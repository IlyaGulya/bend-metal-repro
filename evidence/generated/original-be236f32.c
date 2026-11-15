
// Imports
// =======

// The Objective-C headers take #include: a build (-o) reads an
// #import as the framework of an effect.

#pragma clang fp contract(off)

#ifdef __CUDACC_RTC__
#define BEND_RTC 1
#endif

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#elif !BEND_RTC
#ifdef __APPLE__
#define _DARWIN_UNLIMITED_SELECT
#else
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <time.h>
#include <poll.h>
#include <sys/select.h>
#include <fcntl.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#ifdef __OBJC__
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#elif BEND_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include <sys/stat.h>
#endif
#endif

// Dialect
// =======

// Metal needs coherent(device) (MSL 3.2), or M1-class parts lose stores
// across threadgroups; CUDA lanes hand off through a32 and FENCE. The
// preserve_* attributes need clang 19+ and an optimized build. A segment is
// a case of the device's switch, or a host preserve_none function entered
// by musttail.

#ifdef __METAL_VERSION__
#if __METAL_VERSION__ >= 320
#define DEV     coherent(device) device
#else
#define DEV     device
#endif
#define THR     thread
#define TG      threadgroup
#define INLINE  inline
#define OUTLINE static
#define CONSTV  constant
#define DEVICE  1
#define FENCE() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)
#define BAR()   threadgroup_barrier(mem_flags::mem_threadgroup)
#define BARD()  threadgroup_barrier(mem_flags::mem_device \
  | mem_flags::mem_threadgroup)
#else
#define DEV
#define THR
#define TG
#define INLINE  static inline
#define CONSTV  static const
#ifdef BEND_RTC
#define OUTLINE FAR
#define DEVICE  1
#define clz(x)  (u32)__clz((int)(x))
#define popcount(x) (u32)__popc(x)
#define FENCE() __threadfence()
#define BAR()   __syncthreads()
#define BARD()  { __threadfence(); __syncthreads(); }
#else
#if __has_attribute(preserve_none) && __has_attribute(preserve_most) \
  && __OPTIMIZE__
#define PRESERVE(A) __attribute__((A))
#else
#define PRESERVE(A)
#endif
#define OUTLINE FAR __attribute__((cold)) PRESERVE(preserve_most)
#define DEVICE  0
#define clz(x)  (u32)__builtin_clz(x)
#define popcount(x) (u32)__builtin_popcount(x)
#define FENCE() ((void)0)
#endif
#endif
#define FAR static __attribute__((noinline))

// The device inlines a spin once per caller, so a long one is a call there.
// The host lets clang decide, but for a long F32 loop (emit_native).
#if DEVICE
#define DFAR       FAR
#define LOCK(l)
#define UNLOCK(l)
#define WL_CASE(F) case F: {
#define WL_OPEN    {
#define WL_JMP(F)  { fid = (F); break; }
#define WL_DYN     WL_JMP
#else
#define DFAR       INLINE
#define LOCK(l)    while (__atomic_exchange_n(&(l), 1, __ATOMIC_ACQUIRE)) {}
#define UNLOCK(l)  __atomic_store_n(&(l), 0, __ATOMIC_RELEASE)
#define WL_FN      FAR PRESERVE(preserve_none) Term
#define WL_CASE(F) WL_FN WL_##F(WL_SIG) {
#define WL_OPEN    { WL_BANK u32 rn;
#define WL_JMP(F)  __attribute__((musttail)) return WL_##F(WL_ALL)
#define WL_DYN(F)  __attribute__((musttail)) return wl_tab[F](WL_ALL)
#endif
#define WL_SPIN     for (;;) { if (err_spun(e.mem, &wpoll)) { return 0; }
#define WL_SPUN     } break;
#define WL_AGAIN(F) continue

#define LANE_STEP (DEVICE ? (long)CUBE : 1)
#define STK(I)    sp[(long)(I) * LANE_STEP]

#define WL_RETN(N)  { rn = (N); sp -= LANE_STEP; WL_DYN((u32)STK(0)); }
#define WL_CONT     STK(-3)
#define WL_IDX      STK(-2)
#define WL_POPN(N)  sp -= (N) * LANE_STEP
#define WL_PUSHN(N) sp += (N) * LANE_STEP
#define WL_FRAME(T) \
  u64 wtl = task_tail(T); \
  STK(0) = e.mem[wtl]; \
  STK(1) = (e.mem[wtl + 1] >> 32) & 0xFFFF; \
  STK(2) = FID_EXIT; \
  WL_PUSHN(3);
#define WL_ARGS(A, N) \
  for (u32 wi = 0; wi < (N); wi += 1) { \
    STK(wi) = e.mem[(A) + wi]; \
  } \
  WL_PUSHN(N);
#define WL_ROOM(N) \
  if (DEVICE && sp + (N) * CUBE >= e.mem + STAT_OFF + CUBE) { \
    err_post(e.mem, ERR_DEEP); \
    return 0; \
  }

// Types
// =====

#ifdef __METAL_VERSION__
typedef ulong u64;
typedef uint  u32;
typedef uchar u8;
#elif BEND_RTC
typedef unsigned long long u64;
typedef unsigned int       u32;
typedef unsigned char      u8;
#else
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t  u8;
#endif
typedef unsigned short u16;
typedef float f32;

typedef u64 Term;

typedef struct {
  DEV u64* mem;
  DEV u64* alc;
} Env;

typedef struct {
  u64 off;
  u32 rd;
  u32 wr;
  u32 top;
} Bank;

#if DEVICE
typedef u32 u32a;
#else
typedef u32 __attribute__((may_alias)) u32a;
#endif

// Constants
// =========

#define TAG_PAK 1ull
#define TAG_CTR 2ull
#define TAG_CLO 3ull
#define TAG_BUF 4ull
#define TAG_TSK 5ull
#define TAG_ARR 6ull

#define TERM_HOLE (~0ull)
#define LOC_MASK  ((1ull << 40) - 1)
#define RFC_BIT   (1ull << 63)
#define RFC_CNT   ((1u << 24) - 1)
#define NAT_IMM   ((1ull << 48) - 1)

enum { ERR_RING = 1, ERR_TAGS, ERR_HEAP, ERR_FIDS, ERR_NATS, ERR_RFCS,
  ERR_DEEP, ERR_ARRS, ERR_CNTS };

#define LINE      16
#define PAGE_BITS 7
#define PAGE_LEN  (1ull << PAGE_BITS)
#define CUBE_T    128
#define CUBE      ((u64)CUBE_T * CUBE_T)
#define CUBE_G    (1u << CUBE_LOG)
#define LANES     ((u64)CUBE_T << CUBE_LOG)
#define RING_LEN  (1ull << (17 - CUBE_LOG))
#define STAK_LEN  (1ull << 11)
#define NCLS      8
#define NCLS_ALL  32
#define IO_HELP   64

#define TG_HOLD   2304
#define CHUNK     256
#define CAP_WORDS 32768
#define QUANTUM   (DEVICE ? PAGE_LEN \
  : KEEP_WORDS < 32 * PAGE_LEN ? KEEP_WORDS : 32 * PAGE_LEN)
#if DEVICE
#define KEEP_WORDS CHUNK
#endif
#define RING_WORDS ((1ull << 10) + 2)

#define H_BUMP       0
#define H_CAP        1
#define H_CURSOR     LINE
#define H_FRONT      (H_CURSOR + 1)
#define H_GROUPS     (H_CURSOR + 2)
#define H_ROOT_DONE  (2 * LINE)
#define H_ERROR_CODE (3 * LINE)
#define H_ROOT_WORD  (4 * LINE)
#define H_BANK       (H_ROOT_WORD + WL_RESW)

#define PAGE_UP(n) (((n) + PAGE_LEN - 1) & ~(PAGE_LEN - 1))
#define ALC_OFF  PAGE_UP(H_BANK + 3 * NCLS_ALL)
#define RING_OFF (ALC_OFF + CUBE * 2 * NCLS_ALL)
#define STAK_OFF (RING_OFF + CUBE * RING_WORDS)
#define STAT_OFF (STAK_OFF + CUBE * STAK_LEN)
#define HEAP_OFF (STAT_OFF + PAGE_UP(STAT_LEN))

// Globals
// =======

// The bag is 2^CUBE_LOG groups of CUBE_T lanes (a -D constant on the
// device).

#if !DEVICE

static u64*    CORPUS;
static u64    ALC[CUBE_T][3 * NCLS_ALL] __attribute__((aligned(128)));
static u32    KEEP_WORDS;
static u32    CUBE_LOG = 7;
static u32    bank_lock;

static u32             pool_size;
static u32             pool_row;
static u32             pool_done;
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  pool_wake = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  pool_join = PTHREAD_COND_INITIALIZER;

#if BEND_METAL || BEND_CUDA
#pragma clang diagnostic ignored "-Wc23-extensions"
static const char BEND_SRC[] = {
#embed __FILE__
, 0 };
#endif

#ifdef __OBJC__
static id<MTLDevice>               gpu_dev;
static id<MTLCommandQueue>         gpu_que;
static id<MTLComputePipelineState> gpu_pso;
static id<MTLBuffer>               gpu_buf;
static id<MTLComputeCommandEncoder> gpu_enc;
#elif BEND_CUDA
static CUdevice   gpu_dev;
static CUmodule   gpu_lib;
static CUfunction gpu_pso;
static CUdeviceptr gpu_front;
#endif
static bool io_gpu;
static DEV Term*  io_stk;

static const char* CLI_HELP =
  "usage: %s [+RTS options -RTS] [arguments]\n"
  "  --threads N       worker threads, 1 to 128 (default: the CPU count)\n"
  "  --gpu on|off|4GB  run ! calls on the GPU, over this much of its memory\n"
  "                    (default: on if present, over 2GB on Metal)\n"
  "  --gpu-build       write the GPU program and exit\n"
  "  --help            show this text\n";

#endif

// Tables
// ======

typedef struct { u8 ar, resw, bangs, nofk; } Fid;
typedef struct { u8 ar, hot; u16 spine; } Cid;

#define CID_Tuple 0
#define CID_SNil 1
#define CID_SCon 2
#define CID_WCon 3
#define CID_Emit 4
#define CID_Halt 5
#define CID_Exit 6
#define CID_Fail 7
#define CID_Done 8
#define CID_Ready 9
#define CID_Wait 10
#define CID_None 11
#define CID_Some 12
#define CID_False 13
#define CID_True 14
#define CID_Unit 15
#define CID_Chr 16
#define CID_Nil 17
#define CID_Con 18
#define CID_ir_VI 19
#define CID_ir_VB 20
#define CID_ir_VS 21
#define CID_ir_VSet 22
#define CID_ir_VFun 23
#define CID_ir_VE 24
#define CID_ir_IntDomain 25
#define CID_ir_TreeDomain 26
#define CID_ir_FunctionEmpty 27
#define CID_ir_FunctionNode 28
#define CID_ir_IInt 29
#define CID_ir_IBool 30
#define CID_ir_IStr 31
#define CID_ir_ISlot 32
#define CID_ir_IValue 33
#define CID_ir_ILocal 34
#define CID_ir_ISet 35
#define CID_ir_ISetBinary 36
#define CID_ir_ISetUnary 37
#define CID_ir_IBounded 38
#define CID_ir_IFunction 39
#define CID_ir_IProduct 40
#define CID_ir_ILookup 41
#define CID_ir_IUpdate 42
#define CID_ir_IDomain 43
#define CID_ir_ISequence 44
#define CID_ir_IAdd 45
#define CID_ir_ISub 46
#define CID_ir_INeg 47
#define CID_ir_IEq 48
#define CID_ir_INe 49
#define CID_ir_ILt 50
#define CID_ir_ILe 51
#define CID_ir_IGt 52
#define CID_ir_IGe 53
#define CID_ir_IAnd 54
#define CID_ir_IOr 55
#define CID_ir_INot 56
#define CID_ir_IImp 57
#define CID_ir_IIte 58
#define CID_ir_IBad 59
#define CID_ir_EvalEnv 60
#define CID_ir_DomainEmpty 61
#define CID_ir_DomainNode 62
#define CID_ir_Local 63
#define CID_LT 64
#define CID_EQ 65
#define CID_GT 66
#define CID_ir_DomainBuilt 67
#define CID_ir_FunctionBuilt 68
#define CID_metal_runtime_backend 69
#define CID_IO_args 70
#define CID_metal_runtime_events 71
#define CID_metal_runtime_heap_mib 72
#define CID_IO_print 73
#define FID_ir_sequenceTake 0
#define FID_ir_sequenceTake_k1 1
#define FID_ir_buildFunctionLeft 2
#define FID_ir_buildFunctionLeft_k1 3
#define FID_ir_setAppend 4
#define FID_ir_functionAppendTree 5
#define FID_ir_functionAppendTree_k1 6
#define FID_ir_functionValues 7
#define FID_ir_functionValues_k1 8
#define FID_ir_sequenceValue 9
#define FID_ir_sequenceValue_k1 10
#define FID_ir_sequenceValue_k3 11
#define FID_ir_domainTreeLeft 12
#define FID_ir_domainTreeLeft_k1 13
#define FID_ir_productPrefix 14
#define FID_ir_productPrefix_k1 15
#define FID_ir_functionSequenceKeys 16
#define FID_ir_buildFunctionTree 17
#define FID_ir_buildFunctionTree_k1 18
#define FID_ir_powerAdd 19
#define FID_ir_powerAdd_k1 20
#define FID_ir_powerAdd_k3 21
#define FID_ir_powerAdd_k5 22
#define FID_ir_setMergeLeftEmpty 23
#define FID_List_merge_go_0 24
#define FID_List_merge_go_0_k1 25
#define FID_List_merge_go_0_k3 26
#define FID_List_append 27
#define FID_List_append_k1 28
#define FID_ir_functionOrderStacks 29
#define FID_ir_functionOrderStacks_k1 30
#define FID_ir_functionOrderStacks_k3 31
#define FID_ir_buildDomainTree 32
#define FID_ir_buildDomainTree_k1 33
#define FID_ir_integerSequence 34
#define FID_ir_productColumn 35
#define FID_ir_productColumn_k1 36
#define FID_ir_boundedBoolean 37
#define FID_ir_rangeItems 38
#define FID_ir_rangeItems_k1 39
#define FID_ir_setSubset 40
#define FID_ir_setSubset_k1 41
#define FID_ir_setMerge 42
#define FID_ir_setMerge_k1 43
#define FID_ir_valueLe 44
#define FID_ir_valueLe_k1 45
#define FID_List_sort_pass_0 46
#define FID_List_sort_pass_0_k1 47
#define FID_List_sort_pass_0_k3 48
#define FID_List_concat 49
#define FID_List_concat_k1 50
#define FID_ir_domainSorted 51
#define FID_ir_domainSorted_k1 52
#define FID_ir_domainSorted_k3 53
#define FID_ir_domainSorted_k5 54
#define FID_ir_domainSorted_k7 55
#define FID_ir_functionReplace 56
#define FID_ir_functionReplace_k1 57
#define FID_ir_functionReplace_k3 58
#define FID_ir_functionReplace_k5 59
#define FID_ir_domainPartsOrder 60
#define FID_ir_domainPartsOrder_k1 61
#define FID_ir_productCountFits 62
#define FID_ir_productRows 63
#define FID_ir_productRows_k1 64
#define FID_ir_productFunctions 65
#define FID_ir_productFunctions_k1 66
#define FID_ir_productFunctions_k3 67
#define FID_ir_filteredFinish 68
#define FID_ir_powerCountFits 69
#define FID_ir_powerItems 70
#define FID_ir_powerItems_k1 71
#define FID_ir_powerItems_k3 72
#define FID_ir_setBinaryLists 73
#define FID_ir_setBinaryLists_k1 74
#define FID_ir_domainTreeIndex 75
#define FID_ir_domainTreeIndex_k1 76
#define FID_List_sort_pairs_0 77
#define FID_List_sort_pairs_0_k1 78
#define FID_List_sort_pairs_0_k3 79
#define FID_List_sort_go_0 80
#define FID_List_sort_go_0_k1 81
#define FID_List_length 82
#define FID_List_length_k1 83
#define FID_ir_functionOrder 84
#define FID_ir_listEq 85
#define FID_ir_listEq_k1 86
#define FID_ir_listEq_k3 87
#define FID_ir_valueOrder 88
#define FID_ir_vlen 89
#define FID_ir_vlen_k1 90
#define FID_ir_powerCount 91
#define FID_ir_unionItems 92
#define FID_ir_unionItems_k1 93
#define FID_ir_domainIndex 94
#define FID_List_sort_0 95
#define FID_List_sort_0_k1 96
#define FID_List_sort_0_k3 97
#define FID_ir_setUnique 98
#define FID_ir_setUnique_k1 99
#define FID_ir_veq 100
#define FID_ir_veq_k1 101
#define FID_ir_veq_k3 102
#define FID_ir_functionKeys 103
#define FID_ir_functionKeys_k1 104
#define FID_ir_setSorted 105
#define FID_ir_setSorted_k1 106
#define FID_ir_functionFind 107
#define FID_ir_functionFind_k1 108
#define FID_ir_productCount 109
#define FID_ir_firstErr 110
#define FID_ir_boundedLoop 111
#define FID_ir_boundedLoop_k1 112
#define FID_ir_setBinaryValues 113
#define FID_ir_setBinaryValues_k1 114
#define FID_ir_valueMember 115
#define FID_ir_valueMember_k1 116
#define FID_ir_localFind 117
#define FID_ir_veqb 118
#define FID_ir_veqb_k1 119
#define FID_ir_functionSorted 120
#define FID_ir_functionSorted_k1 121
#define FID_ir_functionSorted_k3 122
#define FID_ir_functionSorted_k5 123
#define FID_ir_functionSorted_k7 124
#define FID_ir_functionSorted_k9 125
#define FID_ir_setBinary 126
#define FID_ir_setBinary_k1 127
#define FID_ir_evalSetItems 128
#define FID_ir_evalSetItems_k1 129
#define FID_ir_evalSetItems_k3 130
#define FID_ir_makeSet 131
#define FID_ir_makeSet_k1 132
#define FID_U32_show_go 133
#define FID_ir_eval 134
#define FID_ir_eval_k1 135
#define FID_ir_eval_k3 136
#define FID_ir_eval_k5 137
#define FID_ir_eval_k7 138
#define FID_ir_eval_k9 139
#define FID_ir_eval_k11 140
#define FID_ir_eval_k13 141
#define FID_ir_eval_k15 142
#define FID_ir_eval_k17 143
#define FID_ir_eval_k19 144
#define FID_ir_eval_k21 145
#define FID_ir_eval_k23 146
#define FID_ir_eval_k25 147
#define FID_ir_eval_k27 148
#define FID_ir_eval_k29 149
#define FID_ir_eval_k31 150
#define FID_ir_eval_k33 151
#define FID_ir_eval_k35 152
#define FID_ir_eval_k37 153
#define FID_ir_eval_k39 154
#define FID_ir_eval_k41 155
#define FID_ir_eval_k43 156
#define FID_ir_eval_k45 157
#define FID_ir_eval_k47 158
#define FID_ir_eval_k49 159
#define FID_ir_eval_k51 160
#define FID_ir_eval_k53 161
#define FID_ir_eval_k55 162
#define FID_ir_eval_k57 163
#define FID_ir_eval_k59 164
#define FID_ir_eval_k61 165
#define FID_ir_eval_k63 166
#define FID_ir_eval_k65 167
#define FID_ir_eval_k67 168
#define FID_ir_eval_k69 169
#define FID_ir_eval_k71 170
#define FID_ir_eval_k73 171
#define FID_ir_eval_k75 172
#define FID_ir_eval_k77 173
#define FID_ir_eval_k79 174
#define FID_ir_eval_k81 175
#define FID_ir_eval_k83 176
#define FID_ir_eval_k85 177
#define FID_ir_eval_k87 178
#define FID_ir_eval_k89 179
#define FID_ir_eval_k91 180
#define FID_ir_eval_k93 181
#define FID_ir_eval_k95 182
#define FID_ir_eval_k97 183
#define FID_ir_eval_k99 184
#define FID_ir_eval_k101 185
#define FID_ir_eval_k103 186
#define FID_ir_eval_k105 187
#define FID_ir_eval_k107 188
#define FID_ir_eval_k109 189
#define FID_ir_eval_k111 190
#define FID_ir_eval_k113 191
#define FID_ir_eval_k115 192
#define FID_ir_eval_k117 193
#define FID_U32_show 194
#define FID_String_append 195
#define FID_String_append_k1 196
#define FID_U32_read_go 197
#define FID_calculate 198
#define FID_U32_read 199
#define FID_IO_bind 200
#define FID_IO_bind_c0 201
#define FID_IO_bind_k2 202
#define FID_main 203
#define FID_main_c0 204
#define FID_main_c1 205
#define FID_main_c2 206
#define FID_main_c3 207
#define FID_main_k5 208
#define FID_main_k7 209
#define FID_main_c8 210
#define FID_main_c9 211
#define FID_main_c10 212
#define FID_main_c11 213
#define FID_main_k13 214
#define FID_main_k15 215
#define FID_main_k17 216
#define FID_main_k19 217
#define FID_main_k21 218
#define FID_main_k23 219
#define FID_main_k25 220
#define FID_main_k27 221
#define FID_main_k29 222
#define FID_main_k31 223
#define FID_main_k33 224
#define FID_main_k35 225
#define FID_main_c36 226
#define FID_main_c37 227
#define FID_main_c38 228
#define FID_main_c39 229
#define FID_main_c40 230
#define FID_metal_runtime_backend 231
#define FID_IO_args 232
#define FID_metal_runtime_events 233
#define FID_metal_runtime_heap_mib 234
#define FID_IO_print 235
#define FID_IO_emit 236
#define FID_Clo_apply 237
#define FID_EXIT 238
#define FID_ENTER 239
CONSTV Fid FID_T[] = { { 2, 0, 0, 1 }, { 2, 1, 0, 1 }, { 4, 0, 0, 1 }, { 6, 3, 0, 1 }, { 3, 0, 0, 1 }, { 3, 0, 0, 1 }, { 4, 1, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 1, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 1, 0, 1 }, { 4, 0, 0, 1 }, { 5, 2, 0, 1 }, { 3, 0, 0, 1 }, { 3, 1, 0, 1 }, { 2, 0, 0, 1 }, { 3, 0, 0, 1 }, { 4, 3, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 2, 1, 0, 1 }, { 1, 1, 0, 1 }, { 4, 0, 0, 1 }, { 4, 0, 0, 1 }, { 2, 1, 0, 1 }, { 7, 1, 0, 1 }, { 2, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 7, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 0, 0, 1 }, { 4, 2, 0, 1 }, { 2, 0, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 10, 0, 0, 1 }, { 2, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 4, 1, 0, 1 }, { 5, 0, 0, 1 }, { 8, 1, 0, 1 }, { 2, 0, 0, 1 }, { 1, 1, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 2, 1, 0, 1 }, { 1, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 4, 1, 0, 1 }, { 2, 2, 0, 1 }, { 2, 2, 0, 1 }, { 2, 2, 0, 1 }, { 3, 0, 0, 1 }, { 7, 1, 0, 1 }, { 4, 1, 0, 1 }, { 4, 1, 0, 1 }, { 3, 0, 0, 1 }, { 3, 1, 0, 1 }, { 4, 0, 0, 1 }, { 1, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 2, 1, 0, 1 }, { 3, 0, 0, 1 }, { 3, 0, 0, 1 }, { 1, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 1, 0, 1 }, { 3, 0, 0, 1 }, { 1, 1, 0, 1 }, { 3, 0, 0, 1 }, { 5, 1, 0, 1 }, { 1, 0, 0, 1 }, { 4, 1, 0, 1 }, { 4, 1, 0, 1 }, { 3, 0, 0, 1 }, { 3, 1, 0, 1 }, { 1, 0, 0, 1 }, { 1, 1, 0, 1 }, { 2, 0, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 1, 0, 1 }, { 2, 0, 0, 1 }, { 2, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 1, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 0, 0, 1 }, { 5, 1, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 1, 1, 0, 1 }, { 2, 0, 0, 1 }, { 3, 1, 0, 1 }, { 2, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 5, 1, 0, 1 }, { 2, 0, 0, 1 }, { 1, 0, 0, 1 }, { 8, 0, 0, 1 }, { 10, 1, 0, 1 }, { 3, 0, 0, 1 }, { 1, 1, 0, 1 }, { 3, 0, 0, 1 }, { 3, 2, 0, 1 }, { 2, 0, 0, 1 }, { 2, 0, 0, 1 }, { 1, 1, 0, 1 }, { 2, 0, 0, 1 }, { 4, 2, 0, 1 }, { 3, 1, 0, 1 }, { 4, 1, 0, 1 }, { 4, 3, 0, 1 }, { 4, 1, 0, 1 }, { 3, 0, 0, 1 }, { 3, 1, 0, 1 }, { 4, 0, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 1, 0, 0, 1 }, { 1, 1, 0, 1 }, { 3, 0, 0, 1 }, { 4, 0, 0, 1 }, { 3, 1, 0, 1 }, { 3, 1, 0, 1 }, { 3, 1, 0, 1 }, { 1, 1, 0, 1 }, { 6, 1, 0, 1 }, { 3, 1, 0, 1 }, { 2, 1, 0, 1 }, { 3, 2, 0, 1 }, { 7, 1, 0, 1 }, { 2, 1, 0, 1 }, { 2, 1, 0, 1 }, { 3, 1, 0, 1 }, { 3, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 2, 2, 0, 1 }, { 7, 1, 0, 1 }, { 7, 1, 0, 1 }, { 10, 2, 0, 1 }, { 4, 1, 0, 1 }, { 2, 1, 0, 1 }, { 1, 1, 0, 1 }, { 2, 1, 0, 1 }, { 2, 1, 0, 1 }, { 4, 2, 0, 1 }, { 2, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, { 2, 2, 0, 1 }, { 3, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 1, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 1, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 1, 0, 1 }, { 2, 1, 0, 1 }, { 5, 1, 0, 1 }, { 1, 1, 0, 1 }, { 5, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, { 5, 1, 0, 1 }, { 1, 1, 0, 1 }, { 6, 1, 0, 1 }, { 1, 0, 0, 1 }, { 2, 0, 0, 1 }, { 2, 1, 0, 1 }, { 2, 0, 0, 1 }, { 1, 0, 1, 1 }, { 1, 0, 0, 1 }, { 3, 0, 0, 0 }, { 3, 0, 0, 0 }, { 2, 1, 0, 0 }, { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 1, 0, 0, 0 }, { 2, 0, 0, 0 }, { 2, 0, 0, 0 }, { 3, 2, 0, 0 }, { 2, 1, 0, 0 }, { 3, 0, 0, 0 }, { 3, 0, 0, 0 }, { 4, 0, 0, 0 }, { 4, 0, 0, 1 }, { 4, 1, 0, 1 }, { 4, 1, 0, 1 }, { 4, 1, 0, 1 }, { 4, 1, 0, 1 }, { 4, 1, 0, 1 }, { 4, 1, 0, 1 }, { 3, 1, 0, 1 }, { 3, 1, 0, 1 }, { 2, 1, 0, 1 }, { 2, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 0, 0, 1 }, { 2, 0, 0, 1 }, { 1, 0, 0, 1 }, { 2, 0, 0, 0 } };
CONSTV Cid CID_T[] = { { 2, 0, 0 }, { 0, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 1, 0, 0 }, { 2, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 1, 0 }, { 0, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 2, 1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, { 2, 1, 0 }, { 2, 1, 1 }, { 1, 1, 0 }, { 2, 1, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 6, 1, 2 }, { 1, 1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, { 3, 1, 0 }, { 2, 1, 0 }, { 4, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 4, 1, 0 }, { 1, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 1, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 2, 1, 0 }, { 1, 1, 0 }, { 2, 1, 0 }, { 3, 1, 0 }, { 1, 1, 0 }, { 2, 1, 0 }, { 0, 1, 0 }, { 4, 1, 1 }, { 2, 1, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 2, 0, 0 }, { 3, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 2, 0, 0 } };
#define STAT_LEN 2605
#define WL_RESW 3
#define BANGS 1
#define WL_BANK Term r0, r1, r2, r3, r4, r5, rp, r6, r7, r8, r9;
#define WL_LOAD(A, N) do { if ((N) <= 0) break; r0 = e.mem[(A) + 0]; if ((N) <= 1) break; r1 = e.mem[(A) + 1]; if ((N) <= 2) break; r2 = e.mem[(A) + 2]; if ((N) <= 3) break; r3 = e.mem[(A) + 3]; if ((N) <= 4) break; r4 = e.mem[(A) + 4]; if ((N) <= 5) break; r5 = e.mem[(A) + 5]; if ((N) <= 6) break; r6 = e.mem[(A) + 6]; if ((N) <= 7) break; r7 = e.mem[(A) + 7]; if ((N) <= 8) break; r8 = e.mem[(A) + 8]; if ((N) <= 9) break; r9 = e.mem[(A) + 9]; } while (0);
#define WL_LAST(X) switch (war) { case 0: r0 = (X); break; case 1: r1 = (X); break; case 2: r2 = (X); break; case 3: r3 = (X); break; case 4: r4 = (X); break; case 5: r5 = (X); break; case 6: r6 = (X); break; case 7: r7 = (X); break; case 8: r8 = (X); break; case 9: r9 = (X); break; }
#define WL_SAVE(V) (V)[0] = r0; (V)[1] = r1; (V)[2] = r2;
#define WL_TAKE(V) r0 = (V)[0]; r1 = (V)[1]; r2 = (V)[2];
#define WL_SIG Env e, DEV Term* sp, u32 seq, u32 rn, Term r0, Term r1, Term r2, Term r3, Term r4, Term r5, Term rp, Term r6, Term r7, Term r8, Term r9
#define WL_ALL e, sp, seq, rn, r0, r1, r2, r3, r4, r5, rp, r6, r7, r8, r9
#define WL_TABLE WL_X(FID_ir_sequenceTake) WL_X(FID_ir_sequenceTake_k1) WL_X(FID_ir_buildFunctionLeft) WL_X(FID_ir_buildFunctionLeft_k1) WL_X(FID_ir_setAppend) WL_X(FID_ir_functionAppendTree) WL_X(FID_ir_functionAppendTree_k1) WL_X(FID_ir_functionValues) WL_X(FID_ir_functionValues_k1) WL_X(FID_ir_sequenceValue) WL_X(FID_ir_sequenceValue_k1) WL_X(FID_ir_sequenceValue_k3) WL_X(FID_ir_domainTreeLeft) WL_X(FID_ir_domainTreeLeft_k1) WL_X(FID_ir_productPrefix) WL_X(FID_ir_productPrefix_k1) WL_X(FID_ir_functionSequenceKeys) WL_X(FID_ir_buildFunctionTree) WL_X(FID_ir_buildFunctionTree_k1) WL_X(FID_ir_powerAdd) WL_X(FID_ir_powerAdd_k1) WL_X(FID_ir_powerAdd_k3) WL_X(FID_ir_powerAdd_k5) WL_X(FID_ir_setMergeLeftEmpty) WL_X(FID_List_merge_go_0) WL_X(FID_List_merge_go_0_k1) WL_X(FID_List_merge_go_0_k3) WL_X(FID_List_append) WL_X(FID_List_append_k1) WL_X(FID_ir_functionOrderStacks) WL_X(FID_ir_functionOrderStacks_k1) WL_X(FID_ir_functionOrderStacks_k3) WL_X(FID_ir_buildDomainTree) WL_X(FID_ir_buildDomainTree_k1) WL_X(FID_ir_integerSequence) WL_X(FID_ir_productColumn) WL_X(FID_ir_productColumn_k1) WL_X(FID_ir_boundedBoolean) WL_X(FID_ir_rangeItems) WL_X(FID_ir_rangeItems_k1) WL_X(FID_ir_setSubset) WL_X(FID_ir_setSubset_k1) WL_X(FID_ir_setMerge) WL_X(FID_ir_setMerge_k1) WL_X(FID_ir_valueLe) WL_X(FID_ir_valueLe_k1) WL_X(FID_List_sort_pass_0) WL_X(FID_List_sort_pass_0_k1) WL_X(FID_List_sort_pass_0_k3) WL_X(FID_List_concat) WL_X(FID_List_concat_k1) WL_X(FID_ir_domainSorted) WL_X(FID_ir_domainSorted_k1) WL_X(FID_ir_domainSorted_k3) WL_X(FID_ir_domainSorted_k5) WL_X(FID_ir_domainSorted_k7) WL_X(FID_ir_functionReplace) WL_X(FID_ir_functionReplace_k1) WL_X(FID_ir_functionReplace_k3) WL_X(FID_ir_functionReplace_k5) WL_X(FID_ir_domainPartsOrder) WL_X(FID_ir_domainPartsOrder_k1) WL_X(FID_ir_productCountFits) WL_X(FID_ir_productRows) WL_X(FID_ir_productRows_k1) WL_X(FID_ir_productFunctions) WL_X(FID_ir_productFunctions_k1) WL_X(FID_ir_productFunctions_k3) WL_X(FID_ir_filteredFinish) WL_X(FID_ir_powerCountFits) WL_X(FID_ir_powerItems) WL_X(FID_ir_powerItems_k1) WL_X(FID_ir_powerItems_k3) WL_X(FID_ir_setBinaryLists) WL_X(FID_ir_setBinaryLists_k1) WL_X(FID_ir_domainTreeIndex) WL_X(FID_ir_domainTreeIndex_k1) WL_X(FID_List_sort_pairs_0) WL_X(FID_List_sort_pairs_0_k1) WL_X(FID_List_sort_pairs_0_k3) WL_X(FID_List_sort_go_0) WL_X(FID_List_sort_go_0_k1) WL_X(FID_List_length) WL_X(FID_List_length_k1) WL_X(FID_ir_functionOrder) WL_X(FID_ir_listEq) WL_X(FID_ir_listEq_k1) WL_X(FID_ir_listEq_k3) WL_X(FID_ir_valueOrder) WL_X(FID_ir_vlen) WL_X(FID_ir_vlen_k1) WL_X(FID_ir_powerCount) WL_X(FID_ir_unionItems) WL_X(FID_ir_unionItems_k1) WL_X(FID_ir_domainIndex) WL_X(FID_List_sort_0) WL_X(FID_List_sort_0_k1) WL_X(FID_List_sort_0_k3) WL_X(FID_ir_setUnique) WL_X(FID_ir_setUnique_k1) WL_X(FID_ir_veq) WL_X(FID_ir_veq_k1) WL_X(FID_ir_veq_k3) WL_X(FID_ir_functionKeys) WL_X(FID_ir_functionKeys_k1) WL_X(FID_ir_setSorted) WL_X(FID_ir_setSorted_k1) WL_X(FID_ir_functionFind) WL_X(FID_ir_functionFind_k1) WL_X(FID_ir_productCount) WL_X(FID_ir_firstErr) WL_X(FID_ir_boundedLoop) WL_X(FID_ir_boundedLoop_k1) WL_X(FID_ir_setBinaryValues) WL_X(FID_ir_setBinaryValues_k1) WL_X(FID_ir_valueMember) WL_X(FID_ir_valueMember_k1) WL_X(FID_ir_localFind) WL_X(FID_ir_veqb) WL_X(FID_ir_veqb_k1) WL_X(FID_ir_functionSorted) WL_X(FID_ir_functionSorted_k1) WL_X(FID_ir_functionSorted_k3) WL_X(FID_ir_functionSorted_k5) WL_X(FID_ir_functionSorted_k7) WL_X(FID_ir_functionSorted_k9) WL_X(FID_ir_setBinary) WL_X(FID_ir_setBinary_k1) WL_X(FID_ir_evalSetItems) WL_X(FID_ir_evalSetItems_k1) WL_X(FID_ir_evalSetItems_k3) WL_X(FID_ir_makeSet) WL_X(FID_ir_makeSet_k1) WL_X(FID_U32_show_go) WL_X(FID_ir_eval) WL_X(FID_ir_eval_k1) WL_X(FID_ir_eval_k3) WL_X(FID_ir_eval_k5) WL_X(FID_ir_eval_k7) WL_X(FID_ir_eval_k9) WL_X(FID_ir_eval_k11) WL_X(FID_ir_eval_k13) WL_X(FID_ir_eval_k15) WL_X(FID_ir_eval_k17) WL_X(FID_ir_eval_k19) WL_X(FID_ir_eval_k21) WL_X(FID_ir_eval_k23) WL_X(FID_ir_eval_k25) WL_X(FID_ir_eval_k27) WL_X(FID_ir_eval_k29) WL_X(FID_ir_eval_k31) WL_X(FID_ir_eval_k33) WL_X(FID_ir_eval_k35) WL_X(FID_ir_eval_k37) WL_X(FID_ir_eval_k39) WL_X(FID_ir_eval_k41) WL_X(FID_ir_eval_k43) WL_X(FID_ir_eval_k45) WL_X(FID_ir_eval_k47) WL_X(FID_ir_eval_k49) WL_X(FID_ir_eval_k51) WL_X(FID_ir_eval_k53) WL_X(FID_ir_eval_k55) WL_X(FID_ir_eval_k57) WL_X(FID_ir_eval_k59) WL_X(FID_ir_eval_k61) WL_X(FID_ir_eval_k63) WL_X(FID_ir_eval_k65) WL_X(FID_ir_eval_k67) WL_X(FID_ir_eval_k69) WL_X(FID_ir_eval_k71) WL_X(FID_ir_eval_k73) WL_X(FID_ir_eval_k75) WL_X(FID_ir_eval_k77) WL_X(FID_ir_eval_k79) WL_X(FID_ir_eval_k81) WL_X(FID_ir_eval_k83) WL_X(FID_ir_eval_k85) WL_X(FID_ir_eval_k87) WL_X(FID_ir_eval_k89) WL_X(FID_ir_eval_k91) WL_X(FID_ir_eval_k93) WL_X(FID_ir_eval_k95) WL_X(FID_ir_eval_k97) WL_X(FID_ir_eval_k99) WL_X(FID_ir_eval_k101) WL_X(FID_ir_eval_k103) WL_X(FID_ir_eval_k105) WL_X(FID_ir_eval_k107) WL_X(FID_ir_eval_k109) WL_X(FID_ir_eval_k111) WL_X(FID_ir_eval_k113) WL_X(FID_ir_eval_k115) WL_X(FID_ir_eval_k117) WL_X(FID_U32_show) WL_X(FID_String_append) WL_X(FID_String_append_k1) WL_X(FID_U32_read_go) WL_X(FID_calculate) WL_X(FID_U32_read) WL_X(FID_IO_bind) WL_X(FID_IO_bind_c0) WL_X(FID_IO_bind_k2) WL_X(FID_main) WL_X(FID_main_c0) WL_X(FID_main_c1) WL_X(FID_main_c2) WL_X(FID_main_c3) WL_X(FID_main_k5) WL_X(FID_main_k7) WL_X(FID_main_c8) WL_X(FID_main_c9) WL_X(FID_main_c10) WL_X(FID_main_c11) WL_X(FID_main_k13) WL_X(FID_main_k15) WL_X(FID_main_k17) WL_X(FID_main_k19) WL_X(FID_main_k21) WL_X(FID_main_k23) WL_X(FID_main_k25) WL_X(FID_main_k27) WL_X(FID_main_k29) WL_X(FID_main_k31) WL_X(FID_main_k33) WL_X(FID_main_k35) WL_X(FID_main_c36) WL_X(FID_main_c37) WL_X(FID_main_c38) WL_X(FID_main_c39) WL_X(FID_main_c40) WL_X(FID_metal_runtime_backend) WL_X(FID_IO_args) WL_X(FID_metal_runtime_events) WL_X(FID_metal_runtime_heap_mib) WL_X(FID_IO_print) WL_X(FID_IO_emit) WL_X(FID_Clo_apply) WL_X(FID_EXIT)
#define MAIN_FID FID_main
#define BLK_SHR 0

#define TAB_AT(T, S, I) T[S < I ? S : I]

#define fid_arity(x) FID_T[x].ar
#define fid_resw(x)  FID_T[x].resw
#define fid_nofk(x)  FID_T[x].nofk
#define cid_arity(x) CID_T[x].ar

// A32
// ===

// C11's atomics on every lane; a device FENCE releases or acquires.
// Metal's loads read through a volatile local, or the M1 and M2 pipeline
// builds die. A weak CAS may fail with the cell still x: a32_cmpx loops.

#define A32_LOOP(k, x) \
  INLINE u32 a32_##k(DEV u32* p, u32 v) { \
    u32 o = a32_load(p); \
    while (!a32_cas(p, &o, x)) {} \
    return o; \
  }

#ifdef __METAL_VERSION__

INLINE DEV atomic_uint* A32(DEV u32* p) {
  return (DEV atomic_uint*)p;
}

INLINE TG atomic_uint* A32(TG u32* p) {
  return (TG atomic_uint*)p;
}

#define a32_load(p) \
  ({ volatile thread u32 _a32v = atomic_load_explicit(A32(p), RLX); _a32v; })

#else

#define a32_load(p) atomic_load_explicit(A32(p), RLX)

#ifdef BEND_RTC

__device__ u64 dev_front;
#define A32(p) (p)
#define atomic_load_explicit(p, o)     (*(volatile u32*)(p))
#define atomic_store_explicit(p, v, o) (*(volatile u32*)(p) = (v))
#define atomic_fetch_add_explicit(p, v, o) atomicAdd((u32*)(p), v)
#define atomic_fetch_sub_explicit(p, v, o) atomicSub((u32*)(p), v)
#define atomic_fetch_and_explicit(p, v, o) atomicAnd((u32*)(p), v)
#define atomic_fetch_or_explicit(p, v, o) atomicOr((u32*)(p), v)
#define atomic_fetch_xor_explicit(p, v, o) atomicXor((u32*)(p), v)
#define atomic_fetch_min_explicit(p, v, o) atomicMin((u32*)(p), v)
#define atomic_fetch_max_explicit(p, v, o) atomicMax((u32*)(p), v)
#define atomic_compare_exchange_weak_explicit(p, e, v, s, f) a32_swp(p, e, v)

INLINE bool a32_swp(DEV u32* p, u32* e, u32 v) {
  u32 x = *e;
  *e = atomicCAS((u32*)p, x, v);
  return *e == x;
}

#else

#define A32(p) ((_Atomic u32*)(p))
#define atomic_fetch_min_explicit __c11_atomic_fetch_min
#define atomic_fetch_max_explicit __c11_atomic_fetch_max

#endif

#endif

#define RLX memory_order_relaxed

#if DEVICE
#define REL RLX
#define ACQ RLX
#define ACR RLX
#define a32_acq(p) FENCE()
#define w64_load(p) (*(p))
#else
#define REL memory_order_release
#define ACQ memory_order_acquire
#define ACR memory_order_acq_rel
#define a32_acq(p) ((void)a32_load_acq(p))
#define w64_load(p) atomic_load_explicit((_Atomic u64*)(p), RLX)
#endif

#define a32_store(p, v)     atomic_store_explicit(A32(p), v, RLX)
#define a32_add(p, v) atomic_fetch_add_explicit(A32(p), v, RLX)
#define a32_sub(p, v) atomic_fetch_sub_explicit(A32(p), v, RLX)
#define a32_and(p, v) atomic_fetch_and_explicit(A32(p), v, RLX)
#define a32_or(p, v) atomic_fetch_or_explicit(A32(p), v, RLX)
#define a32_xor(p, v) atomic_fetch_xor_explicit(A32(p), v, RLX)
#define a32_min(p, v) atomic_fetch_min_explicit(A32(p), v, RLX)
#define a32_max(p, v) atomic_fetch_max_explicit(A32(p), v, RLX)
#define a32_sub_rel(p, v)   (FENCE(), atomic_fetch_sub_explicit(A32(p), v, REL))
#define a32_store_rel(p, v) (FENCE(), atomic_store_explicit(A32(p), v, REL))
#define a32_at(H, word)     ((DEV u32*)&(H)[word])

INLINE u32 a32_load_acq(DEV u32* p) {
  u32 v = DEVICE ? a32_load(p) : atomic_load_explicit(A32(p), ACQ);
  FENCE();
  return v;
}

INLINE bool a32_cas(DEV u32* p, THR u32* e, u32 v) {
  FENCE();
  bool ok = atomic_compare_exchange_weak_explicit(A32(p), e, v, ACR, ACQ);
  FENCE();
  return ok;
}

A32_LOOP(exch, v)

INLINE u32 a32_cmpx(DEV u32* p, u32 x, u32 v) {
  u32 o = x;
  while (!a32_cas(p, &o, v) && o == x) {}
  return o;
}

// Err
// ===

#if DEVICE

INLINE void err_post(DEV u64* H, u32 code) {
  a32_cmpx(a32_at(H, H_ERROR_CODE), 0, code);
}

#else

static const char* ERR_TEXT[] = { "",
  "runtime fail-stop",
  "runtime fail-stop",
  "out of memory: run again with a bigger span, as in +RTS --gpu 8GB",
  "a function the device does not hold",
  "a Nat past the largest immediate 2^48-1",
  "runtime fail-stop",
  "memory fault (machine stack overflow?)",
  "an array past the deepest block class 31 (depth + log2 words per cell)",
  "a value has more than 2^24-1 live copies: keep fewer alive at once, or build it again for some of them (each build counts its own copies)",
  "expected a Nat host number: an integer from 0 to 2^53" };

// First failure wins; others wait a second for it to terminate.
FAR int err_claim(void) {
  static _Atomic uintptr_t first;
  uintptr_t self = (uintptr_t)pthread_self(), seen = 0;
  if (atomic_compare_exchange_strong(&first, &seen, self)) return 1;
  struct timespec t = { seen != self, 0 };
  while (nanosleep(&t, &t)) {}
  return 0;
}

OUTLINE void err_fail(const char* msg) {
  if (err_claim()) fflush(stdout);
  fprintf(stderr, "bend: %s\n", msg);
  _exit(1);
}

static void err_post(u64* H, u32 code) {
  err_fail(ERR_TEXT[code]);
}

// Flush if free, report the fault, then re-raise its signal.
static void err_trap(int sig) {
  static const char line[] = "bend: memory fault (machine stack overflow?)\n";
  if (err_claim() && ftrylockfile(stdout) == 0) fflush(stdout);
  if (write(2, line, sizeof line - 1)) {}
  signal(sig, SIG_DFL);
  raise(sig);
}

#endif

#define err_seen(H)    (DEVICE && a32_load(a32_at(H, H_ERROR_CODE)) != 0)
#define err_spun(H, n) ((++*(n) & 4095) == 0 && err_seen(H))

#if __METAL_VERSION__ >= 320
#define err_peek(H) *(volatile DEV u32*)a32_at(H, H_ERROR_CODE)
#else
#define err_peek err_seen
#endif

#ifdef __METAL_VERSION__
INLINE f32 atan2_c99(f32 y, f32 x) {
  return y == 0.0f && x == x
    ? copysign(signbit(x) ? M_PI_F : 0.0f, y) : atan2(y, x);
}
#define sqrt  precise::sqrt
#define exp   precise::exp
#define log   precise::log
#define log2  precise::log2
#define log10 precise::log10
#define sin   fast::sin
#define cos   fast::cos
#define tan   fast::tan
#define pow   precise::pow
#define fmod  precise::fmod
#define atan2 atan2_c99
#define U32_QUO(a, b) \
  ((a) / 2 / (b) * 2 + ((a) - (a) / 2 / (b) * 2 * (b) >= (b)))
#else
#define U32_QUO(a, b) ((a) / (b))
#endif

#define U32_BIN(a, o, b) ((u64)((u32)(a) o (u32)(b)))

INLINE u64 u32_pow(u32 a, u64 n) {
  u32 r = 1;
  for (; n; n >>= 1, a *= a) {
    if (n & 1) r *= a;
  }
  return r;
}

INLINE f32 f32_unbox(u64 x) {
  union { u32 u; f32 f; } p = { (u32)x };
  return p.f;
}

INLINE u64 f32_rewrap(f32 x) {
  union { f32 f; u32 u; } p = { x };
  return p.u;
}

INLINE u64 f32_to_u32(u64 a) {
  f32 v = f32_unbox(a);
  return v >= 0.0f && v < 4294967296.0f ? (u32)v : 0;
}

INLINE u64 nat_chk(Env e, u64 n) {
  if (n > NAT_IMM) {
    err_post(e.mem, ERR_NATS);
    return NAT_IMM;
  }
  return n;
}

INLINE u64 nat_mul(Env e, u64 a, u64 b) {
  return nat_chk(e, b != 0 && a > NAT_IMM / b ? NAT_IMM + 1 : a * b);
}

INLINE u64 map_msb_u(u64 n, u32 x) {
  u64 w = x ? 32 - clz(x) : 0;
  return n < w ? n : w;
}

#if DEVICE

#define f32_show(e, x) (err_post(e.mem, ERR_FIDS), 0)
#define f32_read(e, s) (err_post(e.mem, ERR_FIDS), 0)

#else

static Term f32_show(Env e, Term x);
static Term f32_read(Env e, Term s);

#endif

A32_LOOP(fadd, f32_rewrap(f32_unbox(o) + f32_unbox(v)))

// Bank
// ====

// A stack of exact generations per class. The host pops and pushes at rd;
// a device pass pops below rd and pushes above top, compacted after it.

#define bank_at(H, c) ((DEV Bank*)((H) + H_BANK) + (c))

INLINE u64 bank_pop(DEV u64* H, u32 c) {
  DEV Bank* b = bank_at(H, c);
  u64 got = 0;
  LOCK(bank_lock);
  u32 t = a32_sub(&b->rd, 1);
  if ((int)t > 0) {
    got = H[b->off + t - 1];
  } else {
    a32_add(&b->rd, 1);
  }
  if (!DEVICE) {
    b->wr = b->top = b->rd;
  }
  UNLOCK(bank_lock);
  return got;
}

INLINE void bank_push(DEV u64* H, u32 c, u64 head) {
  DEV Bank* b = bank_at(H, c);
  LOCK(bank_lock);
  H[b->off + a32_add(&b->wr, 1)] = head;
  if (!DEVICE) {
    b->rd = b->top = b->wr;
  }
  UNLOCK(bank_lock);
}

// Heap
// ====

// Lanes keep a free chain and word count per class. Host chains reaching
// KEEP_WORDS replace COLD and bank old COLD; blocks >= CAP_WORDS >=
// KEEP_WORDS go straight to the bank. A miss takes COLD, the bank or fresh
// memory. Devices bank at dev_cut.

#define ALC_AT(e, i)   (e).alc[(i) * LANE_STEP]
#define ALC_LEN(e, c)  ALC_AT(e, NCLS_ALL + (c))
#define ALC_COLD(e, c) ALC_AT(e, 2 * NCLS_ALL + (c))
#define KEEP(c)        (KEEP_WORDS >> (c) ? KEEP_WORDS >> (c) : 1)

INLINE u32 cls_fit(u32 words) {
  return words > 1 ? 32 - clz(words - 1) : 0;
}

OUTLINE void heap_hand(Env e, u32 cls) {
  bool large = (1ull << cls) >= CAP_WORDS;
  u64 cold = large ? ALC_AT(e, cls) : ALC_COLD(e, cls);
  if (cold) {
    bank_push(e.mem, cls, cold);
  }
  ALC_COLD(e, cls) = large ? 0 : ALC_AT(e, cls);
  ALC_AT(e, cls)   = ALC_LEN(e, cls) = 0;
}

#if DEVICE
#define corpus_grow(H, n) false
#else
static bool corpus_grow(u64* H, u64 need);
#endif

OUTLINE u64 heap_alloc_miss(Env e, u32 cls) {
  DEV u64* H = e.mem;
  u64  got = 0;
  if (!DEVICE) {
    got = ALC_COLD(e, cls);
    ALC_COLD(e, cls) = 0;
  }
  if (!got) {
    got = bank_pop(H, cls);
  }
  u32 n = got ? KEEP(cls) : cls < NCLS ? QUANTUM >> cls : 1;
  if (!got) {
    u32 pages = (n << cls) >> PAGE_BITS;
    u32 p     = a32_add(a32_at(H, H_BUMP), pages);
    if ((u64)p + pages > a32_load_acq(a32_at(H, H_CAP))
      && !corpus_grow(H, (u64)p + pages)) {
      err_post(H, ERR_HEAP);
      return HEAP_OFF;
    }
    got = HEAP_OFF + ((u64)p << PAGE_BITS);
    for (u32 i = 1; i <= n; i += 1) {
      H[got + ((u64)(i - 1) << cls)] = i < n ? got + ((u64)i << cls) : 0;
    }
  }
  ALC_AT(e, cls)  = H[got];
  ALC_LEN(e, cls) = (u64)(n - 1) << cls;
  return got;
}

INLINE u64 heap_alloc(Env e, u32 cls) {
  u64 h = ALC_AT(e, cls);
  if (h) {
    ALC_AT(e, cls)   = e.mem[h];
    ALC_LEN(e, cls) -= 1ull << cls;
    return h;
  }
  return heap_alloc_miss(e, cls);
}

INLINE void heap_free(Env e, u32 cls, u64 loc) {
  if (err_peek(e.mem)) {
    return;
  }
#ifdef BEND_POISON
  for (u64 i = 1; i < 1ull << cls; i += 1) {
    e.mem[loc + i] = RFC_BIT | TAG_CTR << 56 | LOC_MASK;
  }
#endif
  e.mem[loc]       = ALC_AT(e, cls);
  ALC_AT(e, cls)   = loc;
  ALC_LEN(e, cls) += 1ull << cls;
  if (!DEVICE && ALC_LEN(e, cls) >= KEEP_WORDS) {
    heap_hand(e, cls);
  }
}

INLINE void spare_free(Env e, u32 cls, u64 loc) {
  if (loc >= HEAP_OFF) {
    heap_free(e, cls, loc);
  }
}

// Term
// ====

#define term_make(tag, aux, loc) \
  (((u64)(tag) << 56) | ((u64)(aux) << 40) | (u64)(loc))

#define term_ctr(cid, loc) term_make(TAG_CTR, cid, loc)
#define term_pak(cid, loc) term_make(TAG_PAK, cid, loc)
#define term_clo(fid, loc) term_make(TAG_CLO, fid, loc)
#define term_tsk(fid, loc) term_make(TAG_TSK, fid, loc)

INLINE Term term_blk(bool arr, u32 cls, u64 loc) {
  return term_make(arr ? TAG_ARR : TAG_BUF, cls, loc);
}

INLINE u64 term_tag(Term t) {
  return (t >> 56) & 0x7f;
}

INLINE bool term_rfc(Term t) {
  return (t & RFC_BIT) != 0;
}

INLINE u32 term_aux(Term t) {
  return (t >> 40) & 0xFFFF;
}

INLINE u64 term_loc(Term t) {
  return t & LOC_MASK;
}

INLINE bool term_triv(Term t) {
  return term_tag(t) <= TAG_PAK || t == TERM_HOLE || term_loc(t) < HEAP_OFF;
}

OUTLINE Term rfc_wrap(Env e, Term t, u32 cnt) {
  if (term_tag(t) == TAG_CLO || term_tag(t) == TAG_TSK) {
    err_post(e.mem, ERR_RFCS);
    return t;
  }
  u64 r = heap_alloc(e, 0);
  e.mem[r] = (term_loc(t) << 24) | cnt;
  return (t & ~LOC_MASK) | RFC_BIT | r;
}

INLINE Term rfc_seal(Env e, Term t) {
  if (term_tag(t) != TAG_CTR || term_rfc(t)) {
    return t;
  }
  return rfc_wrap(e, t, 1);
}

INLINE void rfc_bump(Env e, u64 r, u32 k) {
  u32 c = a32_add(a32_at(e.mem, r), k);
  if ((c & RFC_CNT) >= RFC_CNT - k) {
    err_post(e.mem, ERR_CNTS);
  }
}

INLINE Term term_keep(Env e, Term t, u32 k) {
  if (term_rfc(t)) {
    rfc_bump(e, term_loc(t), k);
    return t;
  }
  if (term_triv(t)) {
    return t;
  }
  return rfc_wrap(e, t, 1 + k);
}

// A redirect cell holds its target's loc over a 24-bit count, which only
// atomic adds on the low word change: the loc bits never do. The host reads
// the cell in one relaxed 64-bit load; the device's plain load may tear into
// two halves, both holding the same loc. A shared node's fields never change
// either, so ctr_take copies them before it acquires.
INLINE u64 term_peek(DEV u64* H, Term t) {
  if (term_rfc(t)) {
    return w64_load(&H[term_loc(t)]) >> 24;
  }
  return term_loc(t);
}

// Base's Map.bit on the key in place: a char is a presence bit, 32 code bits
INLINE u64 map_bit(Env e, Term key, u64 pos) {
  for (u64 i = pos / 33; term_aux(key) == CID_SCon; i -= 1) {
    u64 at = term_peek(e.mem, key), off = pos % 33;
    if (i == 0) {
      return off == 0 || ((u32)e.mem[at] >> (32 - off) & 1);
    }
    key = e.mem[at + 1];
  }
  return 0;
}

#define blk_shr(t) (BLK_SHR && term_rfc(t))

#define blk_loc(H, a) (BLK_SHR ? term_peek(H, a) : term_loc(a))

INLINE u32 blk_cls(Term t) {
  return term_aux(t) & 31;
}

INLINE u32 blk_wcls(bool arr, u32 c) {
  return arr ? c : c == 0 ? 0 : c - 1;
}

INLINE u32 blk_span(Term t) {
  return blk_wcls(term_tag(t) == TAG_ARR, blk_cls(t));
}

// A node is freed before its last live child, its spine swapped last. An end
// over 255 goes in its trivial last cell.
FAR void term_drop(Env e, Term t) {
  DEV u64* H = e.mem;
  u64  cur = 0;
  Term c0  = 0;
  u32  step = 0;
  for (;;) {
    if (!term_triv(t) && term_rfc(t)) {
      u64      r = term_loc(t);
      DEV u32* p = a32_at(H, r);
      if ((a32_sub_rel(p, 1) & RFC_CNT) != 1) {
        t = 0;
      } else {
        a32_acq(p);
        t = (t & ~(RFC_BIT | LOC_MASK)) | (H[r] >> 24);
        heap_free(e, 0, r);
      }
    }
    if (!term_triv(t)) {
      u64 tag = term_tag(t);
      if (tag == TAG_BUF) {
        heap_free(e, blk_span(t), term_loc(t));
      } else {
        u32 aux = term_aux(t);
        u64 loc = term_loc(t);
        u32 n = tag == TAG_ARR ? 0 : tag == TAG_CTR ? cid_arity(aux)
          : fid_arity(aux) - (tag == TAG_CLO),
          cls = tag == TAG_ARR ? 64 | blk_cls(t)
          : n > 247 ? 64 | (n - 240)
          : cls_fit(tag == TAG_TSK ? n + 2 : n),
          w = cls < 64 ? n : 1u << (cls & 63),
          d = tag == TAG_CTR ? CID_T[aux].spine : 0, m = w;
        if (d) {
          u64 end = loc + w - 1;
          Term f = H[end - d];
          H[end - d] = H[end];
          H[end] = f;
        }
        while (m && term_triv(H[loc + m - 1])) --m;
        if (m < 256) {
          n   = m;
          cls &= 63;
        } else if (m < w) {
          H[loc + w - 1] = m;
        }
        c0 = m ? H[loc] : 0;
        H[loc] = cur;
        cur = loc | ((u64)n << 48) | ((u64)cls << 56);
      }
    }
    for (;;) {
      if (err_spun(H, &step) || cur == 0) {
        return;
      }
      u64  loc = cur & LOC_MASK;
      u32  i   = (u8)(cur >> 40);
      u32  n   = (u8)(cur >> 48);
      u32  cls = (u32)(cur >> 56);
      bool arr = cls > 63;
      u32  j   = i;
      if (arr) {
        cls &= 63;
        n   = 1u << cls;
        Term z = H[loc + (i > 1 ? 1 : n - 1)];
        n = i > 1 ? z >> 32 : term_triv(z) ? (u32)z : n;
        j = i > 1 ? (u32)z : i;
      }
      Term c = j == 0 ? c0 : H[loc + j];
      if (j + 1 >= n) {
        cur = H[loc];
        heap_free(e, cls, loc);
      } else {
        if (arr && j) {
          H[loc + 1] = (u64)n << 32 | (j + 1);
        }
        if (!arr || i < 2) {
          cur += 1ull << 40;
        }
      }
      if (!term_triv(c)) {
        t = c;
        break;
      }
    }
  }
}

INLINE void term_sink(Env e, Term t) {
  if (!term_triv(t)) {
    term_drop(e, t);
  }
}

// Inlined, M1 Metal runs wrong (#1954).
FAR u64 span_fade(Env e, Term t, u64 src, u32 n) {
  DEV u32* c = a32_at(e.mem, term_loc(t));
  if ((a32_load(c) & RFC_CNT) == 1) {
    a32_acq(c);
    heap_free(e, 0, term_loc(t));
    return src;
  }
  for (u32 j = 0; j < n; j += 1) {
    Term f = e.mem[src + j];
    if (term_rfc(f)) {
      rfc_bump(e, term_loc(f), 1);
    } else if (!term_triv(f)) {
      err_post(e.mem, ERR_RFCS);
    }
  }
  term_drop(e, t);
  return 0;
}

#define COPY(D, S, N) for (u64 i_ = 0, n_ = (N); i_ != n_; i_ += 1) (D)[i_] = (S)[i_]

INLINE u64 ctr_take(Env e, Term t, u32 n, THR Term* out) {
  u64 src = term_peek(e.mem, t);
  COPY(out, e.mem + src, n);
  return term_rfc(t) ? span_fade(e, t, src, n) : src;
}

INLINE Term term_word(Env e, Term w) {
  u32 x = 0;
  Term t = w;
  for (u32 i = 0; i < 32 && term_aux(t) == CID_WCon; i += 1) {
    u64 l = term_peek(e.mem, t);
    x |= (u32)(e.mem[l] & 1) << i;
    t = e.mem[l + 1];
  }
  term_sink(e, w);
  return x;
}

// Blk
// ===

// A block owns one allocation in its class (an ARR 2^c Terms, a BUF 2^c
// u32). Matching ANode is blk_half twice (the high call frees the source).

#define BLK_ALLOC(n, w) \
  u64 n = heap_alloc(e, w); \
  if (err_seen(e.mem)) { \
    return term_blk(0, 0, n); \
  }

INLINE DEV u32a* blk_ptr(DEV u64* H, u64 loc, u32 i) {
  return (DEV u32a*)(H + loc) + i;
}

INLINE Term blk_read(DEV u64* H, bool arr, u64 loc, u32 i) {
  if (arr) {
    return H[loc + i];
  }
  return (u64)*blk_ptr(H, loc, i);
}

INLINE void blk_write(DEV u64* H, bool arr, u64 loc, u32 i, Term v) {
  if (arr) {
    H[loc + i] = v;
  } else {
    *blk_ptr(H, loc, i) = (u32)v;
  }
}

INLINE u32 blk_at(Term a, u64 i, u32 lgs) {
  return ((u32)i & (u32)((1ull << (blk_cls(a) - lgs)) - 1)) << lgs;
}

INLINE Term blk_keep(Env e, u64 at) {
  Term w = e.mem[at];
  Term v = term_keep(e, w, 1);
  if (v != w) {
    e.mem[at] = v;
  }
  return v;
}

INLINE void blk_fill(Env e, u64 dst, u64 src, u64 n, bool keep) {
  COPY(e.mem + dst, e.mem + src, n);
  for (u64 j = 0, k; keep && j != n; j += k) {
    Term w = e.mem[dst + j];
    for (k = 1; j + k != n && e.mem[dst + j + k] == w; k += 1) {}
    if ((e.mem[dst + j] = term_keep(e, w, k)) != w) {
      e.mem[src + j] = e.mem[dst + j];
    }
  }
}

INLINE void blk_free(Env e, Term t) {
  blk_shr(t) ? term_drop(e, t) : heap_free(e, blk_span(t), term_loc(t));
}

OUTLINE Term blk_copy(Env e, Term a) {
  bool arr = term_tag(a) == TAG_ARR;
  u32 cls = blk_span(a);
  BLK_ALLOC(dst, cls)
  blk_fill(e, dst, blk_loc(e.mem, a), 1ull << cls, arr);
  return term_blk(arr, blk_cls(a), dst);
}

INLINE Term blk_node(Env e, Term l, Term r) {
  DEV u64* H = e.mem;
  bool arr = term_tag(l) == TAG_ARR;
  u32 c = blk_cls(l);
  if (c != blk_cls(r) || c + 1 >= NCLS_ALL) {
    err_post(H, ERR_TAGS);
    return l;
  }
  u64 pl = blk_loc(H, l);
  u64 pr = blk_loc(H, r);
  BLK_ALLOC(n, arr ? c + 1 : c)
  if (!arr && c == 0) {
    H[n] = (u64)*blk_ptr(H, pl, 0) | ((u64)*blk_ptr(H, pr, 0) << 32);
  } else {
    u64 cw = 1ull << blk_span(l);
    blk_fill(e, n, pl, cw, arr && blk_shr(l));
    blk_fill(e, n + cw, pr, cw, arr && blk_shr(r));
  }
  blk_free(e, l);
  blk_free(e, r);
  return term_blk(arr, c + 1, n);
}

INLINE Term blk_half(Env e, Term a, u32 hi) {
  DEV u64* H = e.mem;
  bool arr = term_tag(a) == TAG_ARR;
  u32 c = blk_cls(a);
  if (c == 0) {
    err_post(H, ERR_TAGS);
    return a;
  }
  c -= 1;
  u32 cw = blk_wcls(arr, c);
  u64 src = blk_loc(H, a);
  BLK_ALLOC(n, cw)
  if (!arr && c == 0) {
    H[n] = (u64)*blk_ptr(H, src, hi);
  } else {
    blk_fill(e, n, src + ((u64)hi << cw), 1ull << cw, arr && blk_shr(a));
  }
  if (hi) {
    blk_free(e, a);
  }
  return term_blk(arr, c, n);
}

INLINE Term blk_new(Env e, bool arr, u64 d, u32 lgs, u32 n, THR Term* v) {
  DEV u64* H = e.mem;
  if (d > 31 - lgs) {
    err_post(H, ERR_ARRS);
    d = 0;
  }
  u32 c = (u32)d + lgs;
  BLK_ALLOC(l, blk_wcls(arr, c))
  for (u32 j = 0; arr && d > 0 && j < n; j += 1) {
    if (d >= 24 && !term_triv(v[j])) {
      err_post(H, ERR_CNTS);
      n = 0;
    }
    v[j] = term_keep(e, v[j], (1u << d) - 1);
  }
  for (u32 i = 0; i != 1u << c; i += 1u << lgs) {
    for (u32 j = 0; j != 1u << lgs; j += 1) {
      blk_write(H, arr, l, i + j, j < n ? v[j] : 0);
    }
  }
  return term_blk(arr, c, l);
}

// Ring
// ====

#define ring_word(H, r, w) ((H) + RING_OFF + (w) * LANES + (r))
#define ring_slot(H, r, p) ring_word(H, r, (p) & (RING_LEN - 1))
#define ring_get(H, r)     ((DEV u32*)ring_word(H, r, RING_LEN))
#define ring_put(H, r)     ((DEV u32*)ring_word(H, r, RING_LEN + 1))
#define ring_held(H, r)    (ring_put(H, r) + 1)

INLINE u32 ring_lap(u32 pos) {
  return ~(u32)(pos / RING_LEN) & 1;
}

INLINE void ring_push(DEV u64* H, u32 r, Term tsk) {
  u32 pos = a32_add(ring_put(H, r), 1);
  if (pos - a32_load(ring_get(H, r)) >= RING_LEN) {
    err_post(H, ERR_RING);
    return;
  }
  DEV u32* lo = (DEV u32*)ring_slot(H, r, pos);
  a32_store(lo, (u32)tsk);
  a32_store_rel(lo + 1, (u32)(tsk >> 32) | (ring_lap(pos) << 31));
}

INLINE u32 ring_flip(u32 i) {
  return (i % CUBE_T << CUBE_LOG) + i / CUBE_T;
}

// A turn's five dispatches (x, y, z) sit at H_GROUPS, before H_ROOT_DONE.
typedef char turn_fits[H_GROUPS + 8 <= H_ROOT_DONE ? 1 : -1];

#define TURN_ROWS(at) ((at) % 4 == 0 ? 1 : CUBE_G)

INLINE u32 turn_groups(u64 f, u32 at) {
  return f != 0 && (at == 2 ? f >= CUBE_T : at == 1 ? f < LANES
    : at == 4 || f < CUBE_T) ? TURN_ROWS(at) : 0;
}

INLINE void turn_set(DEV u64* H, u64 f) {
  H[H_FRONT] = f;
#ifdef BEND_RTC
  dev_front = f;
#endif
  for (u32 i = 0; i < 15; i += 1) {
    ((DEV u32*)(H + H_GROUPS))[i] = i % 3 ? 1 : turn_groups(f, i / 3);
  }
}

#define turn_take(H) a32_exch(a32_at(H, H_CURSOR), 0)

#define ring_pick(b, s, c) ((b) + (s) * (a32_add(c, 1) & (CUBE_T - 1)))

// Task
// ====

INLINE u64 task_node(Env e, u32 fid, Term cont, u32 idx, u32 rem) {
  u32 ar  = fid_arity(fid);
  u64 loc = heap_alloc(e, cls_fit(ar + 2));
  for (u32 i = 0; rem && i < ar; i += 1) {
    e.mem[loc + i] = TERM_HOLE;
  }
  e.mem[loc + ar]     = cont;
  e.mem[loc + ar + 1] = ((u64)idx << 32) | rem;
  return loc;
}

INLINE u64 task_tail(Term t) {
  return term_loc(t) + fid_arity(term_aux(t));
}

INLINE Term task_deliver(DEV u64* H, Term cont, u32 idx, THR Term* v, u32 n) {
  u64 at = cont == TERM_HOLE ? H_ROOT_WORD : term_loc(cont) + idx;
  for (u32 j = 0; j < WL_RESW; j += 1) {
    if (j < n) {
      H[at + j] = v[j];
    }
  }
  if (cont == TERM_HOLE) {
    a32_store_rel(a32_at(H, H_ROOT_DONE), n + 1);
    return 0;
  }
  DEV u32* c = a32_at(H, task_tail(cont) + 1);
  if (a32_sub_rel(c, 1) == 1) {
    a32_acq(c);
    return cont;
  }
  return 0;
}

INLINE void task_deal(DEV u64* H, Term join, u32 base, u32 stride, TG u32* cur) {
  u64 loc = term_loc(join);
  u32 ar  = fid_arity(term_aux(join));
  u32 g   = 0;
  if (stride == 0) {
    g = a32_add(a32_at(H, H_CURSOR), (u32)H[loc + ar + 1]);
  }
  for (u32 i = 0; i < ar; i += 1) {
    Term k = H[loc + i];
    if (term_tag(k) == TAG_TSK) {
      H[loc + i] = TERM_HOLE;
      ring_push(H, stride != 0 ? ring_pick(base, stride, cur)
        : ring_flip(g & (u32)(LANES - 1)), k);
      g += 1;
    }
  }
}

// Root
// ====

INLINE bool root_done(DEV u64* H) {
  return a32_load_acq(a32_at(H, H_ROOT_DONE)) != 0;
}

static u32 root_take(DEV u64* H, THR Term* v) {
  u32 n = a32_load_acq(a32_at(H, H_ROOT_DONE)) - 1;
  COPY(v, H + H_ROOT_WORD, n);
  a32_store(a32_at(H, H_ROOT_DONE), 0);
  return n;
}

// Spins
// =====

CONSTV u64 STAT_IMG[] = { 121ull, term_pak(CID_SNil, 0), 116ull, term_ctr(CID_SCon, STAT_OFF + 0), 105ull, term_ctr(CID_SCon, STAT_OFF + 2), 99ull, term_ctr(CID_SCon, STAT_OFF + 4), 97ull, term_ctr(CID_SCon, STAT_OFF + 6), 112ull, term_ctr(CID_SCon, STAT_OFF + 8), 97ull, term_ctr(CID_SCon, STAT_OFF + 10), 99ull, term_ctr(CID_SCon, STAT_OFF + 12), 32ull, term_ctr(CID_SCon, STAT_OFF + 14), 110ull, term_ctr(CID_SCon, STAT_OFF + 16), 111ull, term_ctr(CID_SCon, STAT_OFF + 18), 105ull, term_ctr(CID_SCon, STAT_OFF + 20), 116ull, term_ctr(CID_SCon, STAT_OFF + 22), 97ull, term_ctr(CID_SCon, STAT_OFF + 24), 99ull, term_ctr(CID_SCon, STAT_OFF + 26), 111ull, term_ctr(CID_SCon, STAT_OFF + 28), 108ull, term_ctr(CID_SCon, STAT_OFF + 30), 108ull, term_ctr(CID_SCon, STAT_OFF + 32), 97ull, term_ctr(CID_SCon, STAT_OFF + 34), 32ull, term_ctr(CID_SCon, STAT_OFF + 36), 100ull, term_ctr(CID_SCon, STAT_OFF + 38), 101ull, term_ctr(CID_SCon, STAT_OFF + 40), 116ull, term_ctr(CID_SCon, STAT_OFF + 42), 114ull, term_ctr(CID_SCon, STAT_OFF + 44), 111ull, term_ctr(CID_SCon, STAT_OFF + 46), 112ull, term_ctr(CID_SCon, STAT_OFF + 48), 112ull, term_ctr(CID_SCon, STAT_OFF + 50), 117ull, term_ctr(CID_SCon, STAT_OFF + 52), 115ull, term_ctr(CID_SCon, STAT_OFF + 54), 32ull, term_ctr(CID_SCon, STAT_OFF + 56), 115ull, term_ctr(CID_SCon, STAT_OFF + 58), 100ull, term_ctr(CID_SCon, STAT_OFF + 60), 101ull, term_ctr(CID_SCon, STAT_OFF + 62), 101ull, term_ctr(CID_SCon, STAT_OFF + 64), 99ull, term_ctr(CID_SCon, STAT_OFF + 66), 120ull, term_ctr(CID_SCon, STAT_OFF + 68), 101ull, term_ctr(CID_SCon, STAT_OFF + 70), 32ull, term_ctr(CID_SCon, STAT_OFF + 72), 101ull, term_ctr(CID_SCon, STAT_OFF + 74), 99ull, term_ctr(CID_SCon, STAT_OFF + 76), 110ull, term_ctr(CID_SCon, STAT_OFF + 78), 101ull, term_ctr(CID_SCon, STAT_OFF + 80), 117ull, term_ctr(CID_SCon, STAT_OFF + 82), 113ull, term_ctr(CID_SCon, STAT_OFF + 84), 101ull, term_ctr(CID_SCon, STAT_OFF + 86), 115ull, term_ctr(CID_SCon, STAT_OFF + 88), term_ctr(CID_SCon, STAT_OFF + 90), 78ull, term_pak(CID_SNil, 0), 73ull, term_ctr(CID_SCon, STAT_OFF + 93), 65ull, term_ctr(CID_SCon, STAT_OFF + 95), 77ull, term_ctr(CID_SCon, STAT_OFF + 97), 79ull, term_ctr(CID_SCon, STAT_OFF + 99), 68ull, term_ctr(CID_SCon, STAT_OFF + 101), 32ull, term_ctr(CID_SCon, STAT_OFF + 103), 101ull, term_ctr(CID_SCon, STAT_OFF + 105), 100ull, term_ctr(CID_SCon, STAT_OFF + 107), 105ull, term_ctr(CID_SCon, STAT_OFF + 109), 115ull, term_ctr(CID_SCon, STAT_OFF + 111), 116ull, term_ctr(CID_SCon, STAT_OFF + 113), 117ull, term_ctr(CID_SCon, STAT_OFF + 115), 111ull, term_ctr(CID_SCon, STAT_OFF + 117), 32ull, term_ctr(CID_SCon, STAT_OFF + 119), 115ull, term_ctr(CID_SCon, STAT_OFF + 121), 105ull, term_ctr(CID_SCon, STAT_OFF + 123), 32ull, term_ctr(CID_SCon, STAT_OFF + 125), 116ull, term_ctr(CID_SCon, STAT_OFF + 127), 110ull, term_ctr(CID_SCon, STAT_OFF + 129), 105ull, term_ctr(CID_SCon, STAT_OFF + 131), 111ull, term_ctr(CID_SCon, STAT_OFF + 133), 112ull, term_ctr(CID_SCon, STAT_OFF + 135), 100ull, term_ctr(CID_SCon, STAT_OFF + 137), 110ull, term_ctr(CID_SCon, STAT_OFF + 139), 101ull, term_ctr(CID_SCon, STAT_OFF + 141), 32ull, term_ctr(CID_SCon, STAT_OFF + 143), 113ull, term_ctr(CID_SCon, STAT_OFF + 145), 101ull, term_ctr(CID_SCon, STAT_OFF + 147), 83ull, term_ctr(CID_SCon, STAT_OFF + 149), 98ull, term_ctr(CID_SCon, STAT_OFF + 151), 117ull, term_ctr(CID_SCon, STAT_OFF + 153), 83ull, term_ctr(CID_SCon, STAT_OFF + 155), term_ctr(CID_SCon, STAT_OFF + 157), term_pak(CID_ir_FunctionEmpty, 0), 1, 110ull, term_pak(CID_SNil, 0), 111ull, term_ctr(CID_SCon, STAT_OFF + 162), 105ull, term_ctr(CID_SCon, STAT_OFF + 164), 116ull, term_ctr(CID_SCon, STAT_OFF + 166), 99ull, term_ctr(CID_SCon, STAT_OFF + 168), 110ull, term_ctr(CID_SCon, STAT_OFF + 170), 117ull, term_ctr(CID_SCon, STAT_OFF + 172), 102ull, term_ctr(CID_SCon, STAT_OFF + 174), 32ull, term_ctr(CID_SCon, STAT_OFF + 176), 101ull, term_ctr(CID_SCon, STAT_OFF + 178), 116ull, term_ctr(CID_SCon, STAT_OFF + 180), 105ull, term_ctr(CID_SCon, STAT_OFF + 182), 110ull, term_ctr(CID_SCon, STAT_OFF + 184), 105ull, term_ctr(CID_SCon, STAT_OFF + 186), 102ull, term_ctr(CID_SCon, STAT_OFF + 188), 32ull, term_ctr(CID_SCon, STAT_OFF + 190), 100ull, term_ctr(CID_SCon, STAT_OFF + 192), 101ull, term_ctr(CID_SCon, STAT_OFF + 194), 114ull, term_ctr(CID_SCon, STAT_OFF + 196), 101ull, term_ctr(CID_SCon, STAT_OFF + 198), 119ull, term_ctr(CID_SCon, STAT_OFF + 200), 111ull, term_ctr(CID_SCon, STAT_OFF + 202), 108ull, term_ctr(CID_SCon, STAT_OFF + 204), 32ull, term_ctr(CID_SCon, STAT_OFF + 206), 100ull, term_ctr(CID_SCon, STAT_OFF + 208), 101ull, term_ctr(CID_SCon, STAT_OFF + 210), 109ull, term_ctr(CID_SCon, STAT_OFF + 212), 114ull, term_ctr(CID_SCon, STAT_OFF + 214), 111ull, term_ctr(CID_SCon, STAT_OFF + 216), 102ull, term_ctr(CID_SCon, STAT_OFF + 218), 108ull, term_ctr(CID_SCon, STAT_OFF + 220), 97ull, term_ctr(CID_SCon, STAT_OFF + 222), 109ull, term_ctr(CID_SCon, STAT_OFF + 224), term_ctr(CID_SCon, STAT_OFF + 226), 116ull, term_pak(CID_SNil, 0), 110ull, term_ctr(CID_SCon, STAT_OFF + 229), 101ull, term_ctr(CID_SCon, STAT_OFF + 231), 109ull, term_ctr(CID_SCon, STAT_OFF + 233), 101ull, term_ctr(CID_SCon, STAT_OFF + 235), 108ull, term_ctr(CID_SCon, STAT_OFF + 237), 101ull, term_ctr(CID_SCon, STAT_OFF + 239), 32ull, term_ctr(CID_SCon, STAT_OFF + 241), 116ull, term_ctr(CID_SCon, STAT_OFF + 243), 101ull, term_ctr(CID_SCon, STAT_OFF + 245), 115ull, term_ctr(CID_SCon, STAT_OFF + 247), 45ull, term_ctr(CID_SCon, STAT_OFF + 249), 114ull, term_ctr(CID_SCon, STAT_OFF + 251), 101ull, term_ctr(CID_SCon, STAT_OFF + 253), 119ull, term_ctr(CID_SCon, STAT_OFF + 255), 111ull, term_ctr(CID_SCon, STAT_OFF + 257), 112ull, term_ctr(CID_SCon, STAT_OFF + 259), 32ull, term_ctr(CID_SCon, STAT_OFF + 261), 100ull, term_ctr(CID_SCon, STAT_OFF + 263), 101ull, term_ctr(CID_SCon, STAT_OFF + 265), 114ull, term_ctr(CID_SCon, STAT_OFF + 267), 101ull, term_ctr(CID_SCon, STAT_OFF + 269), 119ull, term_ctr(CID_SCon, STAT_OFF + 271), 111ull, term_ctr(CID_SCon, STAT_OFF + 273), 108ull, term_ctr(CID_SCon, STAT_OFF + 275), 32ull, term_ctr(CID_SCon, STAT_OFF + 277), 100ull, term_ctr(CID_SCon, STAT_OFF + 279), 105ull, term_ctr(CID_SCon, STAT_OFF + 281), 108ull, term_ctr(CID_SCon, STAT_OFF + 283), 97ull, term_ctr(CID_SCon, STAT_OFF + 285), 118ull, term_ctr(CID_SCon, STAT_OFF + 287), 110ull, term_ctr(CID_SCon, STAT_OFF + 289), 105ull, term_ctr(CID_SCon, STAT_OFF + 291), term_ctr(CID_SCon, STAT_OFF + 293), 101ull, term_pak(CID_SNil, 0), 99ull, term_ctr(CID_SCon, STAT_OFF + 296), 110ull, term_ctr(CID_SCon, STAT_OFF + 298), 101ull, term_ctr(CID_SCon, STAT_OFF + 300), 117ull, term_ctr(CID_SCon, STAT_OFF + 302), 113ull, term_ctr(CID_SCon, STAT_OFF + 304), 101ull, term_ctr(CID_SCon, STAT_OFF + 306), 115ull, term_ctr(CID_SCon, STAT_OFF + 308), 32ull, term_ctr(CID_SCon, STAT_OFF + 310), 121ull, term_ctr(CID_SCon, STAT_OFF + 312), 116ull, term_ctr(CID_SCon, STAT_OFF + 314), 112ull, term_ctr(CID_SCon, STAT_OFF + 316), 109ull, term_ctr(CID_SCon, STAT_OFF + 318), 101ull, term_ctr(CID_SCon, STAT_OFF + 320), 110ull, term_ctr(CID_SCon, STAT_OFF + 322), 111ull, term_ctr(CID_SCon, STAT_OFF + 324), 110ull, term_ctr(CID_SCon, STAT_OFF + 326), 32ull, term_ctr(CID_SCon, STAT_OFF + 328), 97ull, term_ctr(CID_SCon, STAT_OFF + 330), 32ull, term_ctr(CID_SCon, STAT_OFF + 332), 115ull, term_ctr(CID_SCon, STAT_OFF + 334), 101ull, term_ctr(CID_SCon, STAT_OFF + 336), 114ull, term_ctr(CID_SCon, STAT_OFF + 338), 105ull, term_ctr(CID_SCon, STAT_OFF + 340), 117ull, term_ctr(CID_SCon, STAT_OFF + 342), 113ull, term_ctr(CID_SCon, STAT_OFF + 344), 101ull, term_ctr(CID_SCon, STAT_OFF + 346), 114ull, term_ctr(CID_SCon, STAT_OFF + 348), 32ull, term_ctr(CID_SCon, STAT_OFF + 350), 108ull, term_ctr(CID_SCon, STAT_OFF + 352), 105ull, term_ctr(CID_SCon, STAT_OFF + 354), 97ull, term_ctr(CID_SCon, STAT_OFF + 356), 84ull, term_ctr(CID_SCon, STAT_OFF + 358), term_ctr(CID_SCon, STAT_OFF + 360), 110ull, term_ctr(CID_SCon, STAT_OFF + 74), 111ull, term_ctr(CID_SCon, STAT_OFF + 363), 105ull, term_ctr(CID_SCon, STAT_OFF + 365), 116ull, term_ctr(CID_SCon, STAT_OFF + 367), 99ull, term_ctr(CID_SCon, STAT_OFF + 369), 110ull, term_ctr(CID_SCon, STAT_OFF + 371), 117ull, term_ctr(CID_SCon, STAT_OFF + 373), 102ull, term_ctr(CID_SCon, STAT_OFF + 375), 32ull, term_ctr(CID_SCon, STAT_OFF + 377), 101ull, term_ctr(CID_SCon, STAT_OFF + 379), 116ull, term_ctr(CID_SCon, STAT_OFF + 381), 105ull, term_ctr(CID_SCon, STAT_OFF + 383), 110ull, term_ctr(CID_SCon, STAT_OFF + 385), 105ull, term_ctr(CID_SCon, STAT_OFF + 387), 102ull, term_ctr(CID_SCon, STAT_OFF + 389), term_ctr(CID_SCon, STAT_OFF + 391), 97ull, term_ctr(CID_SCon, STAT_OFF + 162), 101ull, term_ctr(CID_SCon, STAT_OFF + 394), 108ull, term_ctr(CID_SCon, STAT_OFF + 396), 111ull, term_ctr(CID_SCon, STAT_OFF + 398), 111ull, term_ctr(CID_SCon, STAT_OFF + 400), 98ull, term_ctr(CID_SCon, STAT_OFF + 402), 32ull, term_ctr(CID_SCon, STAT_OFF + 404), 101ull, term_ctr(CID_SCon, STAT_OFF + 406), 98ull, term_ctr(CID_SCon, STAT_OFF + 408), 32ull, term_ctr(CID_SCon, STAT_OFF + 410), 116ull, term_ctr(CID_SCon, STAT_OFF + 412), 115ull, term_ctr(CID_SCon, STAT_OFF + 414), 117ull, term_ctr(CID_SCon, STAT_OFF + 416), 109ull, term_ctr(CID_SCon, STAT_OFF + 418), 32ull, term_ctr(CID_SCon, STAT_OFF + 420), 101ull, term_ctr(CID_SCon, STAT_OFF + 422), 116ull, term_ctr(CID_SCon, STAT_OFF + 424), 97ull, term_ctr(CID_SCon, STAT_OFF + 426), 99ull, term_ctr(CID_SCon, STAT_OFF + 428), 105ull, term_ctr(CID_SCon, STAT_OFF + 430), 100ull, term_ctr(CID_SCon, STAT_OFF + 432), 101ull, term_ctr(CID_SCon, STAT_OFF + 434), 114ull, term_ctr(CID_SCon, STAT_OFF + 436), 112ull, term_ctr(CID_SCon, STAT_OFF + 438), 32ull, term_ctr(CID_SCon, STAT_OFF + 440), 100ull, term_ctr(CID_SCon, STAT_OFF + 442), 101ull, term_ctr(CID_SCon, STAT_OFF + 444), 100ull, term_ctr(CID_SCon, STAT_OFF + 446), 110ull, term_ctr(CID_SCon, STAT_OFF + 448), 117ull, term_ctr(CID_SCon, STAT_OFF + 450), 111ull, term_ctr(CID_SCon, STAT_OFF + 452), 98ull, term_ctr(CID_SCon, STAT_OFF + 454), term_ctr(CID_SCon, STAT_OFF + 456), 117ull, term_ctr(CID_SCon, STAT_OFF + 296), 108ull, term_ctr(CID_SCon, STAT_OFF + 459), 97ull, term_ctr(CID_SCon, STAT_OFF + 461), 118ull, term_ctr(CID_SCon, STAT_OFF + 463), 32ull, term_ctr(CID_SCon, STAT_OFF + 465), 116ull, term_ctr(CID_SCon, STAT_OFF + 467), 101ull, term_ctr(CID_SCon, STAT_OFF + 469), 115ull, term_ctr(CID_SCon, STAT_OFF + 471), 45ull, term_ctr(CID_SCon, STAT_OFF + 473), 114ull, term_ctr(CID_SCon, STAT_OFF + 475), 101ull, term_ctr(CID_SCon, STAT_OFF + 477), 119ull, term_ctr(CID_SCon, STAT_OFF + 479), 111ull, term_ctr(CID_SCon, STAT_OFF + 481), 112ull, term_ctr(CID_SCon, STAT_OFF + 483), 32ull, term_ctr(CID_SCon, STAT_OFF + 485), 100ull, term_ctr(CID_SCon, STAT_OFF + 487), 101ull, term_ctr(CID_SCon, STAT_OFF + 489), 114ull, term_ctr(CID_SCon, STAT_OFF + 491), 101ull, term_ctr(CID_SCon, STAT_OFF + 493), 119ull, term_ctr(CID_SCon, STAT_OFF + 495), 111ull, term_ctr(CID_SCon, STAT_OFF + 497), 108ull, term_ctr(CID_SCon, STAT_OFF + 499), 32ull, term_ctr(CID_SCon, STAT_OFF + 501), 100ull, term_ctr(CID_SCon, STAT_OFF + 503), 105ull, term_ctr(CID_SCon, STAT_OFF + 505), 108ull, term_ctr(CID_SCon, STAT_OFF + 507), 97ull, term_ctr(CID_SCon, STAT_OFF + 509), 118ull, term_ctr(CID_SCon, STAT_OFF + 511), 110ull, term_ctr(CID_SCon, STAT_OFF + 513), 105ull, term_ctr(CID_SCon, STAT_OFF + 515), term_ctr(CID_SCon, STAT_OFF + 517), 97ull, term_ctr(CID_SCon, STAT_OFF + 467), 32ull, term_ctr(CID_SCon, STAT_OFF + 520), 100ull, term_ctr(CID_SCon, STAT_OFF + 522), 110ull, term_ctr(CID_SCon, STAT_OFF + 524), 97ull, term_ctr(CID_SCon, STAT_OFF + 526), 32ull, term_ctr(CID_SCon, STAT_OFF + 528), 101ull, term_ctr(CID_SCon, STAT_OFF + 530), 108ull, term_ctr(CID_SCon, STAT_OFF + 532), 112ull, term_ctr(CID_SCon, STAT_OFF + 534), 117ull, term_ctr(CID_SCon, STAT_OFF + 536), 116ull, term_ctr(CID_SCon, STAT_OFF + 538), 32ull, term_ctr(CID_SCon, STAT_OFF + 540), 97ull, term_ctr(CID_SCon, STAT_OFF + 542), 32ull, term_ctr(CID_SCon, STAT_OFF + 544), 115ull, term_ctr(CID_SCon, STAT_OFF + 546), 101ull, term_ctr(CID_SCon, STAT_OFF + 548), 114ull, term_ctr(CID_SCon, STAT_OFF + 550), 105ull, term_ctr(CID_SCon, STAT_OFF + 552), 117ull, term_ctr(CID_SCon, STAT_OFF + 554), 113ull, term_ctr(CID_SCon, STAT_OFF + 556), 101ull, term_ctr(CID_SCon, STAT_OFF + 558), 114ull, term_ctr(CID_SCon, STAT_OFF + 560), 32ull, term_ctr(CID_SCon, STAT_OFF + 562), 100ull, term_ctr(CID_SCon, STAT_OFF + 564), 110ull, term_ctr(CID_SCon, STAT_OFF + 566), 101ull, term_ctr(CID_SCon, STAT_OFF + 568), 112ull, term_ctr(CID_SCon, STAT_OFF + 570), 112ull, term_ctr(CID_SCon, STAT_OFF + 572), 65ull, term_ctr(CID_SCon, STAT_OFF + 574), term_ctr(CID_SCon, STAT_OFF + 576), 115ull, term_pak(CID_SNil, 0), 116ull, term_ctr(CID_SCon, STAT_OFF + 579), 110ull, term_ctr(CID_SCon, STAT_OFF + 581), 105ull, term_ctr(CID_SCon, STAT_OFF + 583), 111ull, term_ctr(CID_SCon, STAT_OFF + 585), 112ull, term_ctr(CID_SCon, STAT_OFF + 587), 100ull, term_ctr(CID_SCon, STAT_OFF + 589), 110ull, term_ctr(CID_SCon, STAT_OFF + 591), 101ull, term_ctr(CID_SCon, STAT_OFF + 593), 32ull, term_ctr(CID_SCon, STAT_OFF + 595), 114ull, term_ctr(CID_SCon, STAT_OFF + 597), 101ull, term_ctr(CID_SCon, STAT_OFF + 599), 103ull, term_ctr(CID_SCon, STAT_OFF + 601), 101ull, term_ctr(CID_SCon, STAT_OFF + 603), 116ull, term_ctr(CID_SCon, STAT_OFF + 605), 110ull, term_ctr(CID_SCon, STAT_OFF + 607), 105ull, term_ctr(CID_SCon, STAT_OFF + 609), 32ull, term_ctr(CID_SCon, STAT_OFF + 611), 100ull, term_ctr(CID_SCon, STAT_OFF + 613), 110ull, term_ctr(CID_SCon, STAT_OFF + 615), 97ull, term_ctr(CID_SCon, STAT_OFF + 617), 32ull, term_ctr(CID_SCon, STAT_OFF + 619), 101ull, term_ctr(CID_SCon, STAT_OFF + 621), 108ull, term_ctr(CID_SCon, STAT_OFF + 623), 112ull, term_ctr(CID_SCon, STAT_OFF + 625), 117ull, term_ctr(CID_SCon, STAT_OFF + 627), 116ull, term_ctr(CID_SCon, STAT_OFF + 629), 32ull, term_ctr(CID_SCon, STAT_OFF + 631), 97ull, term_ctr(CID_SCon, STAT_OFF + 633), 32ull, term_ctr(CID_SCon, STAT_OFF + 635), 115ull, term_ctr(CID_SCon, STAT_OFF + 637), 101ull, term_ctr(CID_SCon, STAT_OFF + 639), 114ull, term_ctr(CID_SCon, STAT_OFF + 641), 105ull, term_ctr(CID_SCon, STAT_OFF + 643), 117ull, term_ctr(CID_SCon, STAT_OFF + 645), 113ull, term_ctr(CID_SCon, STAT_OFF + 647), 101ull, term_ctr(CID_SCon, STAT_OFF + 649), 114ull, term_ctr(CID_SCon, STAT_OFF + 651), 32ull, term_ctr(CID_SCon, STAT_OFF + 653), 113ull, term_ctr(CID_SCon, STAT_OFF + 655), 101ull, term_ctr(CID_SCon, STAT_OFF + 657), 83ull, term_ctr(CID_SCon, STAT_OFF + 659), 98ull, term_ctr(CID_SCon, STAT_OFF + 661), 117ull, term_ctr(CID_SCon, STAT_OFF + 663), 83ull, term_ctr(CID_SCon, STAT_OFF + 665), term_ctr(CID_SCon, STAT_OFF + 667), 108ull, term_ctr(CID_SCon, STAT_OFF + 296), 112ull, term_ctr(CID_SCon, STAT_OFF + 670), 117ull, term_ctr(CID_SCon, STAT_OFF + 672), 116ull, term_ctr(CID_SCon, STAT_OFF + 674), 32ull, term_ctr(CID_SCon, STAT_OFF + 676), 97ull, term_ctr(CID_SCon, STAT_OFF + 678), 32ull, term_ctr(CID_SCon, STAT_OFF + 680), 115ull, term_ctr(CID_SCon, STAT_OFF + 682), 101ull, term_ctr(CID_SCon, STAT_OFF + 684), 114ull, term_ctr(CID_SCon, STAT_OFF + 686), 105ull, term_ctr(CID_SCon, STAT_OFF + 688), 117ull, term_ctr(CID_SCon, STAT_OFF + 690), 113ull, term_ctr(CID_SCon, STAT_OFF + 692), 101ull, term_ctr(CID_SCon, STAT_OFF + 694), 114ull, term_ctr(CID_SCon, STAT_OFF + 696), 32ull, term_ctr(CID_SCon, STAT_OFF + 698), 100ull, term_ctr(CID_SCon, STAT_OFF + 700), 97ull, term_ctr(CID_SCon, STAT_OFF + 702), 101ull, term_ctr(CID_SCon, STAT_OFF + 704), 72ull, term_ctr(CID_SCon, STAT_OFF + 706), term_ctr(CID_SCon, STAT_OFF + 708), 108ull, term_ctr(CID_SCon, STAT_OFF + 700), 105ull, term_ctr(CID_SCon, STAT_OFF + 711), 97ull, term_ctr(CID_SCon, STAT_OFF + 713), 84ull, term_ctr(CID_SCon, STAT_OFF + 715), term_ctr(CID_SCon, STAT_OFF + 717), 101ull, term_ctr(CID_SCon, STAT_OFF + 579), 108ull, term_ctr(CID_SCon, STAT_OFF + 720), 112ull, term_ctr(CID_SCon, STAT_OFF + 722), 117ull, term_ctr(CID_SCon, STAT_OFF + 724), 116ull, term_ctr(CID_SCon, STAT_OFF + 726), 32ull, term_ctr(CID_SCon, STAT_OFF + 728), 111ull, term_ctr(CID_SCon, STAT_OFF + 730), 119ull, term_ctr(CID_SCon, STAT_OFF + 732), 116ull, term_ctr(CID_SCon, STAT_OFF + 734), 32ull, term_ctr(CID_SCon, STAT_OFF + 736), 115ull, term_ctr(CID_SCon, STAT_OFF + 738), 101ull, term_ctr(CID_SCon, STAT_OFF + 740), 114ull, term_ctr(CID_SCon, STAT_OFF + 742), 105ull, term_ctr(CID_SCon, STAT_OFF + 744), 117ull, term_ctr(CID_SCon, STAT_OFF + 746), 113ull, term_ctr(CID_SCon, STAT_OFF + 748), 101ull, term_ctr(CID_SCon, STAT_OFF + 750), 114ull, term_ctr(CID_SCon, STAT_OFF + 752), 32ull, term_ctr(CID_SCon, STAT_OFF + 754), 110ull, term_ctr(CID_SCon, STAT_OFF + 756), 111ull, term_ctr(CID_SCon, STAT_OFF + 758), 105ull, term_ctr(CID_SCon, STAT_OFF + 760), 116ull, term_ctr(CID_SCon, STAT_OFF + 762), 97ull, term_ctr(CID_SCon, STAT_OFF + 764), 110ull, term_ctr(CID_SCon, STAT_OFF + 766), 101ull, term_ctr(CID_SCon, STAT_OFF + 768), 116ull, term_ctr(CID_SCon, STAT_OFF + 770), 97ull, term_ctr(CID_SCon, STAT_OFF + 772), 99ull, term_ctr(CID_SCon, STAT_OFF + 774), 110ull, term_ctr(CID_SCon, STAT_OFF + 776), 111ull, term_ctr(CID_SCon, STAT_OFF + 778), 99ull, term_ctr(CID_SCon, STAT_OFF + 780), term_ctr(CID_SCon, STAT_OFF + 782), 110ull, term_ctr(CID_SCon, STAT_OFF + 700), 101ull, term_ctr(CID_SCon, STAT_OFF + 785), 76ull, term_ctr(CID_SCon, STAT_OFF + 787), term_ctr(CID_SCon, STAT_OFF + 789), term_pak(CID_Nil, 0), term_pak(CID_Nil, 0), term_pak(CID_ir_DomainEmpty, 0), term_pak(CID_Nil, 0), term_ctr(CID_ir_TreeDomain, STAT_OFF + 794), term_ctr(CID_ir_VSet, STAT_OFF + 795), term_pak(CID_Nil, 0), 103ull, term_ctr(CID_SCon, STAT_OFF + 76), 110ull, term_ctr(CID_SCon, STAT_OFF + 799), 97ull, term_ctr(CID_SCon, STAT_OFF + 801), 114ull, term_ctr(CID_SCon, STAT_OFF + 803), 32ull, term_ctr(CID_SCon, STAT_OFF + 805), 101ull, term_ctr(CID_SCon, STAT_OFF + 807), 116ull, term_ctr(CID_SCon, STAT_OFF + 809), 105ull, term_ctr(CID_SCon, STAT_OFF + 811), 110ull, term_ctr(CID_SCon, STAT_OFF + 813), 105ull, term_ctr(CID_SCon, STAT_OFF + 815), 102ull, term_ctr(CID_SCon, STAT_OFF + 817), term_ctr(CID_SCon, STAT_OFF + 819), 114ull, term_pak(CID_SNil, 0), 111ull, term_ctr(CID_SCon, STAT_OFF + 822), 116ull, term_ctr(CID_SCon, STAT_OFF + 824), 97ull, term_ctr(CID_SCon, STAT_OFF + 826), 114ull, term_ctr(CID_SCon, STAT_OFF + 828), 101ull, term_ctr(CID_SCon, STAT_OFF + 830), 112ull, term_ctr(CID_SCon, STAT_OFF + 832), 111ull, term_ctr(CID_SCon, STAT_OFF + 834), 32ull, term_ctr(CID_SCon, STAT_OFF + 836), 116ull, term_ctr(CID_SCon, STAT_OFF + 838), 101ull, term_ctr(CID_SCon, STAT_OFF + 840), 115ull, term_ctr(CID_SCon, STAT_OFF + 842), 32ull, term_ctr(CID_SCon, STAT_OFF + 844), 100ull, term_ctr(CID_SCon, STAT_OFF + 846), 101ull, term_ctr(CID_SCon, STAT_OFF + 848), 114ull, term_ctr(CID_SCon, STAT_OFF + 850), 101ull, term_ctr(CID_SCon, STAT_OFF + 852), 119ull, term_ctr(CID_SCon, STAT_OFF + 854), 111ull, term_ctr(CID_SCon, STAT_OFF + 856), 108ull, term_ctr(CID_SCon, STAT_OFF + 858), 32ull, term_ctr(CID_SCon, STAT_OFF + 860), 110ull, term_ctr(CID_SCon, STAT_OFF + 862), 119ull, term_ctr(CID_SCon, STAT_OFF + 864), 111ull, term_ctr(CID_SCon, STAT_OFF + 866), 110ull, term_ctr(CID_SCon, STAT_OFF + 868), 107ull, term_ctr(CID_SCon, STAT_OFF + 870), 110ull, term_ctr(CID_SCon, STAT_OFF + 872), 117ull, term_ctr(CID_SCon, STAT_OFF + 874), term_ctr(CID_SCon, STAT_OFF + 876), 101ull, term_ctr(CID_SCon, STAT_OFF + 838), 99ull, term_ctr(CID_SCon, STAT_OFF + 879), 110ull, term_ctr(CID_SCon, STAT_OFF + 881), 101ull, term_ctr(CID_SCon, STAT_OFF + 883), 117ull, term_ctr(CID_SCon, STAT_OFF + 885), 113ull, term_ctr(CID_SCon, STAT_OFF + 887), 101ull, term_ctr(CID_SCon, STAT_OFF + 889), 115ull, term_ctr(CID_SCon, STAT_OFF + 891), 32ull, term_ctr(CID_SCon, STAT_OFF + 893), 100ull, term_ctr(CID_SCon, STAT_OFF + 895), 101ull, term_ctr(CID_SCon, STAT_OFF + 897), 114ull, term_ctr(CID_SCon, STAT_OFF + 899), 101ull, term_ctr(CID_SCon, STAT_OFF + 901), 119ull, term_ctr(CID_SCon, STAT_OFF + 903), 111ull, term_ctr(CID_SCon, STAT_OFF + 905), 108ull, term_ctr(CID_SCon, STAT_OFF + 907), 32ull, term_ctr(CID_SCon, STAT_OFF + 909), 110ull, term_ctr(CID_SCon, STAT_OFF + 911), 119ull, term_ctr(CID_SCon, STAT_OFF + 913), 111ull, term_ctr(CID_SCon, STAT_OFF + 915), 110ull, term_ctr(CID_SCon, STAT_OFF + 917), 107ull, term_ctr(CID_SCon, STAT_OFF + 919), 110ull, term_ctr(CID_SCon, STAT_OFF + 921), 117ull, term_ctr(CID_SCon, STAT_OFF + 923), term_ctr(CID_SCon, STAT_OFF + 925), 116ull, term_ctr(CID_SCon, STAT_OFF + 74), 101ull, term_ctr(CID_SCon, STAT_OFF + 928), 115ull, term_ctr(CID_SCon, STAT_OFF + 930), 32ull, term_ctr(CID_SCon, STAT_OFF + 932), 101ull, term_ctr(CID_SCon, STAT_OFF + 934), 116ull, term_ctr(CID_SCon, STAT_OFF + 936), 105ull, term_ctr(CID_SCon, STAT_OFF + 938), 110ull, term_ctr(CID_SCon, STAT_OFF + 940), 105ull, term_ctr(CID_SCon, STAT_OFF + 942), 102ull, term_ctr(CID_SCon, STAT_OFF + 944), term_ctr(CID_SCon, STAT_OFF + 946), 110ull, term_ctr(CID_SCon, STAT_OFF + 934), 111ull, term_ctr(CID_SCon, STAT_OFF + 949), 105ull, term_ctr(CID_SCon, STAT_OFF + 951), 116ull, term_ctr(CID_SCon, STAT_OFF + 953), 99ull, term_ctr(CID_SCon, STAT_OFF + 955), 110ull, term_ctr(CID_SCon, STAT_OFF + 957), 117ull, term_ctr(CID_SCon, STAT_OFF + 959), 102ull, term_ctr(CID_SCon, STAT_OFF + 961), 32ull, term_ctr(CID_SCon, STAT_OFF + 963), 101ull, term_ctr(CID_SCon, STAT_OFF + 965), 116ull, term_ctr(CID_SCon, STAT_OFF + 967), 105ull, term_ctr(CID_SCon, STAT_OFF + 969), 110ull, term_ctr(CID_SCon, STAT_OFF + 971), 105ull, term_ctr(CID_SCon, STAT_OFF + 973), 102ull, term_ctr(CID_SCon, STAT_OFF + 975), term_ctr(CID_SCon, STAT_OFF + 977), 100ull, term_ctr(CID_SCon, STAT_OFF + 838), 101ull, term_ctr(CID_SCon, STAT_OFF + 980), 100ull, term_ctr(CID_SCon, STAT_OFF + 982), 110ull, term_ctr(CID_SCon, STAT_OFF + 984), 117ull, term_ctr(CID_SCon, STAT_OFF + 986), 111ull, term_ctr(CID_SCon, STAT_OFF + 988), 98ull, term_ctr(CID_SCon, STAT_OFF + 990), 32ull, term_ctr(CID_SCon, STAT_OFF + 992), 100ull, term_ctr(CID_SCon, STAT_OFF + 994), 101ull, term_ctr(CID_SCon, STAT_OFF + 996), 114ull, term_ctr(CID_SCon, STAT_OFF + 998), 101ull, term_ctr(CID_SCon, STAT_OFF + 1000), 119ull, term_ctr(CID_SCon, STAT_OFF + 1002), 111ull, term_ctr(CID_SCon, STAT_OFF + 1004), 108ull, term_ctr(CID_SCon, STAT_OFF + 1006), 32ull, term_ctr(CID_SCon, STAT_OFF + 1008), 110ull, term_ctr(CID_SCon, STAT_OFF + 1010), 119ull, term_ctr(CID_SCon, STAT_OFF + 1012), 111ull, term_ctr(CID_SCon, STAT_OFF + 1014), 110ull, term_ctr(CID_SCon, STAT_OFF + 1016), 107ull, term_ctr(CID_SCon, STAT_OFF + 1018), 110ull, term_ctr(CID_SCon, STAT_OFF + 1020), 117ull, term_ctr(CID_SCon, STAT_OFF + 1022), term_ctr(CID_SCon, STAT_OFF + 1024), 114ull, term_ctr(CID_SCon, STAT_OFF + 934), 101ull, term_ctr(CID_SCon, STAT_OFF + 1027), 119ull, term_ctr(CID_SCon, STAT_OFF + 1029), 111ull, term_ctr(CID_SCon, STAT_OFF + 1031), 112ull, term_ctr(CID_SCon, STAT_OFF + 1033), term_ctr(CID_SCon, STAT_OFF + 1035), 100ull, term_pak(CID_SNil, 0), 110ull, term_ctr(CID_SCon, STAT_OFF + 1038), 97ull, term_ctr(CID_SCon, STAT_OFF + 1040), 114ull, term_ctr(CID_SCon, STAT_OFF + 1042), 101ull, term_ctr(CID_SCon, STAT_OFF + 1044), 112ull, term_ctr(CID_SCon, STAT_OFF + 1046), 111ull, term_ctr(CID_SCon, STAT_OFF + 1048), 32ull, term_ctr(CID_SCon, STAT_OFF + 1050), 110ull, term_ctr(CID_SCon, STAT_OFF + 1052), 97ull, term_ctr(CID_SCon, STAT_OFF + 1054), 101ull, term_ctr(CID_SCon, STAT_OFF + 1056), 108ull, term_ctr(CID_SCon, STAT_OFF + 1058), 111ull, term_ctr(CID_SCon, STAT_OFF + 1060), 111ull, term_ctr(CID_SCon, STAT_OFF + 1062), 98ull, term_ctr(CID_SCon, STAT_OFF + 1064), 32ull, term_ctr(CID_SCon, STAT_OFF + 1066), 97ull, term_ctr(CID_SCon, STAT_OFF + 1068), 32ull, term_ctr(CID_SCon, STAT_OFF + 1070), 100ull, term_ctr(CID_SCon, STAT_OFF + 1072), 101ull, term_ctr(CID_SCon, STAT_OFF + 1074), 116ull, term_ctr(CID_SCon, STAT_OFF + 1076), 99ull, term_ctr(CID_SCon, STAT_OFF + 1078), 101ull, term_ctr(CID_SCon, STAT_OFF + 1080), 112ull, term_ctr(CID_SCon, STAT_OFF + 1082), 120ull, term_ctr(CID_SCon, STAT_OFF + 1084), 101ull, term_ctr(CID_SCon, STAT_OFF + 1086), term_ctr(CID_SCon, STAT_OFF + 1088), 119ull, term_pak(CID_SNil, 0), 111ull, term_ctr(CID_SCon, STAT_OFF + 1091), 108ull, term_ctr(CID_SCon, STAT_OFF + 1093), 102ull, term_ctr(CID_SCon, STAT_OFF + 1095), 114ull, term_ctr(CID_SCon, STAT_OFF + 1097), 101ull, term_ctr(CID_SCon, STAT_OFF + 1099), 118ull, term_ctr(CID_SCon, STAT_OFF + 1101), 111ull, term_ctr(CID_SCon, STAT_OFF + 1103), 32ull, term_ctr(CID_SCon, STAT_OFF + 1105), 110ull, term_ctr(CID_SCon, STAT_OFF + 1107), 111ull, term_ctr(CID_SCon, STAT_OFF + 1109), 105ull, term_ctr(CID_SCon, STAT_OFF + 1111), 116ull, term_ctr(CID_SCon, STAT_OFF + 1113), 97ull, term_ctr(CID_SCon, STAT_OFF + 1115), 103ull, term_ctr(CID_SCon, STAT_OFF + 1117), 101ull, term_ctr(CID_SCon, STAT_OFF + 1119), 110ull, term_ctr(CID_SCon, STAT_OFF + 1121), 32ull, term_ctr(CID_SCon, STAT_OFF + 1123), 116ull, term_ctr(CID_SCon, STAT_OFF + 1125), 105ull, term_ctr(CID_SCon, STAT_OFF + 1127), 98ull, term_ctr(CID_SCon, STAT_OFF + 1129), 45ull, term_ctr(CID_SCon, STAT_OFF + 1131), 50ull, term_ctr(CID_SCon, STAT_OFF + 1133), 51ull, term_ctr(CID_SCon, STAT_OFF + 1135), 32ull, term_ctr(CID_SCon, STAT_OFF + 1137), 100ull, term_ctr(CID_SCon, STAT_OFF + 1139), 101ull, term_ctr(CID_SCon, STAT_OFF + 1141), 110ull, term_ctr(CID_SCon, STAT_OFF + 1143), 103ull, term_ctr(CID_SCon, STAT_OFF + 1145), 105ull, term_ctr(CID_SCon, STAT_OFF + 1147), 115ull, term_ctr(CID_SCon, STAT_OFF + 1149), term_ctr(CID_SCon, STAT_OFF + 1151), 99ull, term_ctr(CID_SCon, STAT_OFF + 1115), 97ull, term_ctr(CID_SCon, STAT_OFF + 1154), 114ull, term_ctr(CID_SCon, STAT_OFF + 1156), 116ull, term_ctr(CID_SCon, STAT_OFF + 1158), 98ull, term_ctr(CID_SCon, STAT_OFF + 1160), 117ull, term_ctr(CID_SCon, STAT_OFF + 1162), 115ull, term_ctr(CID_SCon, STAT_OFF + 1164), 32ull, term_ctr(CID_SCon, STAT_OFF + 1166), 116ull, term_ctr(CID_SCon, STAT_OFF + 1168), 105ull, term_ctr(CID_SCon, STAT_OFF + 1170), 98ull, term_ctr(CID_SCon, STAT_OFF + 1172), 45ull, term_ctr(CID_SCon, STAT_OFF + 1174), 50ull, term_ctr(CID_SCon, STAT_OFF + 1176), 51ull, term_ctr(CID_SCon, STAT_OFF + 1178), 32ull, term_ctr(CID_SCon, STAT_OFF + 1180), 100ull, term_ctr(CID_SCon, STAT_OFF + 1182), 101ull, term_ctr(CID_SCon, STAT_OFF + 1184), 110ull, term_ctr(CID_SCon, STAT_OFF + 1186), 103ull, term_ctr(CID_SCon, STAT_OFF + 1188), 105ull, term_ctr(CID_SCon, STAT_OFF + 1190), 115ull, term_ctr(CID_SCon, STAT_OFF + 1192), term_ctr(CID_SCon, STAT_OFF + 1194), 105ull, term_ctr(CID_SCon, STAT_OFF + 1115), 100ull, term_ctr(CID_SCon, STAT_OFF + 1197), 100ull, term_ctr(CID_SCon, STAT_OFF + 1199), 97ull, term_ctr(CID_SCon, STAT_OFF + 1201), 32ull, term_ctr(CID_SCon, STAT_OFF + 1203), 116ull, term_ctr(CID_SCon, STAT_OFF + 1205), 105ull, term_ctr(CID_SCon, STAT_OFF + 1207), 98ull, term_ctr(CID_SCon, STAT_OFF + 1209), 45ull, term_ctr(CID_SCon, STAT_OFF + 1211), 50ull, term_ctr(CID_SCon, STAT_OFF + 1213), 51ull, term_ctr(CID_SCon, STAT_OFF + 1215), 32ull, term_ctr(CID_SCon, STAT_OFF + 1217), 100ull, term_ctr(CID_SCon, STAT_OFF + 1219), 101ull, term_ctr(CID_SCon, STAT_OFF + 1221), 110ull, term_ctr(CID_SCon, STAT_OFF + 1223), 103ull, term_ctr(CID_SCon, STAT_OFF + 1225), 105ull, term_ctr(CID_SCon, STAT_OFF + 1227), 115ull, term_ctr(CID_SCon, STAT_OFF + 1229), term_ctr(CID_SCon, STAT_OFF + 1231), 121ull, term_ctr(CID_SCon, STAT_OFF + 127), 101ull, term_ctr(CID_SCon, STAT_OFF + 1234), 107ull, term_ctr(CID_SCon, STAT_OFF + 1236), 32ull, term_ctr(CID_SCon, STAT_OFF + 1238), 110ull, term_ctr(CID_SCon, STAT_OFF + 1240), 111ull, term_ctr(CID_SCon, STAT_OFF + 1242), 105ull, term_ctr(CID_SCon, STAT_OFF + 1244), 116ull, term_ctr(CID_SCon, STAT_OFF + 1246), 99ull, term_ctr(CID_SCon, STAT_OFF + 1248), 110ull, term_ctr(CID_SCon, STAT_OFF + 1250), 117ull, term_ctr(CID_SCon, STAT_OFF + 1252), 102ull, term_ctr(CID_SCon, STAT_OFF + 1254), term_ctr(CID_SCon, STAT_OFF + 1256), 101ull, term_ctr(CID_SCon, STAT_OFF + 581), 115ull, term_ctr(CID_SCon, STAT_OFF + 1259), 32ull, term_ctr(CID_SCon, STAT_OFF + 1261), 101ull, term_ctr(CID_SCon, STAT_OFF + 1263), 116ull, term_ctr(CID_SCon, STAT_OFF + 1265), 105ull, term_ctr(CID_SCon, STAT_OFF + 1267), 110ull, term_ctr(CID_SCon, STAT_OFF + 1269), 105ull, term_ctr(CID_SCon, STAT_OFF + 1271), 102ull, term_ctr(CID_SCon, STAT_OFF + 1273), 32ull, term_ctr(CID_SCon, STAT_OFF + 1275), 101ull, term_ctr(CID_SCon, STAT_OFF + 1277), 98ull, term_ctr(CID_SCon, STAT_OFF + 1279), 32ull, term_ctr(CID_SCon, STAT_OFF + 1281), 116ull, term_ctr(CID_SCon, STAT_OFF + 1283), 115ull, term_ctr(CID_SCon, STAT_OFF + 1285), 117ull, term_ctr(CID_SCon, STAT_OFF + 1287), 109ull, term_ctr(CID_SCon, STAT_OFF + 1289), 32ull, term_ctr(CID_SCon, STAT_OFF + 1291), 115ull, term_ctr(CID_SCon, STAT_OFF + 1293), 101ull, term_ctr(CID_SCon, STAT_OFF + 1295), 103ull, term_ctr(CID_SCon, STAT_OFF + 1297), 110ull, term_ctr(CID_SCon, STAT_OFF + 1299), 97ull, term_ctr(CID_SCon, STAT_OFF + 1301), 114ull, term_ctr(CID_SCon, STAT_OFF + 1303), 32ull, term_ctr(CID_SCon, STAT_OFF + 1305), 110ull, term_ctr(CID_SCon, STAT_OFF + 1307), 111ull, term_ctr(CID_SCon, STAT_OFF + 1309), 105ull, term_ctr(CID_SCon, STAT_OFF + 1311), 116ull, term_ctr(CID_SCon, STAT_OFF + 1313), 99ull, term_ctr(CID_SCon, STAT_OFF + 1315), 110ull, term_ctr(CID_SCon, STAT_OFF + 1317), 117ull, term_ctr(CID_SCon, STAT_OFF + 1319), 102ull, term_ctr(CID_SCon, STAT_OFF + 1321), 32ull, term_ctr(CID_SCon, STAT_OFF + 1323), 101ull, term_ctr(CID_SCon, STAT_OFF + 1325), 116ull, term_ctr(CID_SCon, STAT_OFF + 1327), 105ull, term_ctr(CID_SCon, STAT_OFF + 1329), 110ull, term_ctr(CID_SCon, STAT_OFF + 1331), 105ull, term_ctr(CID_SCon, STAT_OFF + 1333), 102ull, term_ctr(CID_SCon, STAT_OFF + 1335), term_ctr(CID_SCon, STAT_OFF + 1337), 99ull, term_ctr(CID_SCon, STAT_OFF + 229), 117ull, term_ctr(CID_SCon, STAT_OFF + 1340), 100ull, term_ctr(CID_SCon, STAT_OFF + 1342), 111ull, term_ctr(CID_SCon, STAT_OFF + 1344), 114ull, term_ctr(CID_SCon, STAT_OFF + 1346), 112ull, term_ctr(CID_SCon, STAT_OFF + 1348), 32ull, term_ctr(CID_SCon, STAT_OFF + 1350), 110ull, term_ctr(CID_SCon, STAT_OFF + 1352), 111ull, term_ctr(CID_SCon, STAT_OFF + 1354), 105ull, term_ctr(CID_SCon, STAT_OFF + 1356), 116ull, term_ctr(CID_SCon, STAT_OFF + 1358), 99ull, term_ctr(CID_SCon, STAT_OFF + 1360), 110ull, term_ctr(CID_SCon, STAT_OFF + 1362), 117ull, term_ctr(CID_SCon, STAT_OFF + 1364), 102ull, term_ctr(CID_SCon, STAT_OFF + 1366), 32ull, term_ctr(CID_SCon, STAT_OFF + 1368), 101ull, term_ctr(CID_SCon, STAT_OFF + 1370), 116ull, term_ctr(CID_SCon, STAT_OFF + 1372), 105ull, term_ctr(CID_SCon, STAT_OFF + 1374), 110ull, term_ctr(CID_SCon, STAT_OFF + 1376), 105ull, term_ctr(CID_SCon, STAT_OFF + 1378), 102ull, term_ctr(CID_SCon, STAT_OFF + 1380), 32ull, term_ctr(CID_SCon, STAT_OFF + 1382), 100ull, term_ctr(CID_SCon, STAT_OFF + 1384), 101ull, term_ctr(CID_SCon, STAT_OFF + 1386), 109ull, term_ctr(CID_SCon, STAT_OFF + 1388), 114ull, term_ctr(CID_SCon, STAT_OFF + 1390), 111ull, term_ctr(CID_SCon, STAT_OFF + 1392), 102ull, term_ctr(CID_SCon, STAT_OFF + 1394), 108ull, term_ctr(CID_SCon, STAT_OFF + 1396), 97ull, term_ctr(CID_SCon, STAT_OFF + 1398), 109ull, term_ctr(CID_SCon, STAT_OFF + 1400), term_ctr(CID_SCon, STAT_OFF + 1402), 121ull, term_ctr(CID_SCon, STAT_OFF + 846), 114ull, term_ctr(CID_SCon, STAT_OFF + 1405), 97ull, term_ctr(CID_SCon, STAT_OFF + 1407), 110ull, term_ctr(CID_SCon, STAT_OFF + 1409), 117ull, term_ctr(CID_SCon, STAT_OFF + 1411), 32ull, term_ctr(CID_SCon, STAT_OFF + 1413), 100ull, term_ctr(CID_SCon, STAT_OFF + 1415), 101ull, term_ctr(CID_SCon, STAT_OFF + 1417), 114ull, term_ctr(CID_SCon, STAT_OFF + 1419), 101ull, term_ctr(CID_SCon, STAT_OFF + 1421), 119ull, term_ctr(CID_SCon, STAT_OFF + 1423), 111ull, term_ctr(CID_SCon, STAT_OFF + 1425), 108ull, term_ctr(CID_SCon, STAT_OFF + 1427), 32ull, term_ctr(CID_SCon, STAT_OFF + 1429), 110ull, term_ctr(CID_SCon, STAT_OFF + 1431), 119ull, term_ctr(CID_SCon, STAT_OFF + 1433), 111ull, term_ctr(CID_SCon, STAT_OFF + 1435), 110ull, term_ctr(CID_SCon, STAT_OFF + 1437), 107ull, term_ctr(CID_SCon, STAT_OFF + 1439), 110ull, term_ctr(CID_SCon, STAT_OFF + 1441), 117ull, term_ctr(CID_SCon, STAT_OFF + 1443), term_ctr(CID_SCon, STAT_OFF + 1445), 114ull, term_ctr(CID_SCon, STAT_OFF + 579), 101ull, term_ctr(CID_SCon, STAT_OFF + 1448), 103ull, term_ctr(CID_SCon, STAT_OFF + 1450), 101ull, term_ctr(CID_SCon, STAT_OFF + 1452), 116ull, term_ctr(CID_SCon, STAT_OFF + 1454), 110ull, term_ctr(CID_SCon, STAT_OFF + 1456), 105ull, term_ctr(CID_SCon, STAT_OFF + 1458), 32ull, term_ctr(CID_SCon, STAT_OFF + 1460), 101ull, term_ctr(CID_SCon, STAT_OFF + 1462), 98ull, term_ctr(CID_SCon, STAT_OFF + 1464), 32ull, term_ctr(CID_SCon, STAT_OFF + 1466), 116ull, term_ctr(CID_SCon, STAT_OFF + 1468), 115ull, term_ctr(CID_SCon, STAT_OFF + 1470), 117ull, term_ctr(CID_SCon, STAT_OFF + 1472), 109ull, term_ctr(CID_SCon, STAT_OFF + 1474), 32ull, term_ctr(CID_SCon, STAT_OFF + 1476), 115ull, term_ctr(CID_SCon, STAT_OFF + 1478), 100ull, term_ctr(CID_SCon, STAT_OFF + 1480), 110ull, term_ctr(CID_SCon, STAT_OFF + 1482), 117ull, term_ctr(CID_SCon, STAT_OFF + 1484), 111ull, term_ctr(CID_SCon, STAT_OFF + 1486), 98ull, term_ctr(CID_SCon, STAT_OFF + 1488), 32ull, term_ctr(CID_SCon, STAT_OFF + 1490), 108ull, term_ctr(CID_SCon, STAT_OFF + 1492), 97ull, term_ctr(CID_SCon, STAT_OFF + 1494), 118ull, term_ctr(CID_SCon, STAT_OFF + 1496), 114ull, term_ctr(CID_SCon, STAT_OFF + 1498), 101ull, term_ctr(CID_SCon, STAT_OFF + 1500), 116ull, term_ctr(CID_SCon, STAT_OFF + 1502), 110ull, term_ctr(CID_SCon, STAT_OFF + 1504), 105ull, term_ctr(CID_SCon, STAT_OFF + 1506), 32ull, term_ctr(CID_SCon, STAT_OFF + 1508), 114ull, term_ctr(CID_SCon, STAT_OFF + 1510), 101ull, term_ctr(CID_SCon, STAT_OFF + 1512), 103ull, term_ctr(CID_SCon, STAT_OFF + 1514), 101ull, term_ctr(CID_SCon, STAT_OFF + 1516), 116ull, term_ctr(CID_SCon, STAT_OFF + 1518), 110ull, term_ctr(CID_SCon, STAT_OFF + 1520), 105ull, term_ctr(CID_SCon, STAT_OFF + 1522), term_ctr(CID_SCon, STAT_OFF + 1524), 100ull, term_ctr(CID_SCon, STAT_OFF + 579), 110ull, term_ctr(CID_SCon, STAT_OFF + 1527), 97ull, term_ctr(CID_SCon, STAT_OFF + 1529), 114ull, term_ctr(CID_SCon, STAT_OFF + 1531), 101ull, term_ctr(CID_SCon, STAT_OFF + 1533), 112ull, term_ctr(CID_SCon, STAT_OFF + 1535), 111ull, term_ctr(CID_SCon, STAT_OFF + 1537), 32ull, term_ctr(CID_SCon, STAT_OFF + 1539), 116ull, term_ctr(CID_SCon, STAT_OFF + 1541), 101ull, term_ctr(CID_SCon, STAT_OFF + 1543), 115ull, term_ctr(CID_SCon, STAT_OFF + 1545), 32ull, term_ctr(CID_SCon, STAT_OFF + 1547), 101ull, term_ctr(CID_SCon, STAT_OFF + 1549), 116ull, term_ctr(CID_SCon, STAT_OFF + 1551), 105ull, term_ctr(CID_SCon, STAT_OFF + 1553), 110ull, term_ctr(CID_SCon, STAT_OFF + 1555), 105ull, term_ctr(CID_SCon, STAT_OFF + 1557), 102ull, term_ctr(CID_SCon, STAT_OFF + 1559), 32ull, term_ctr(CID_SCon, STAT_OFF + 1561), 101ull, term_ctr(CID_SCon, STAT_OFF + 1563), 114ull, term_ctr(CID_SCon, STAT_OFF + 1565), 105ull, term_ctr(CID_SCon, STAT_OFF + 1567), 117ull, term_ctr(CID_SCon, STAT_OFF + 1569), 113ull, term_ctr(CID_SCon, STAT_OFF + 1571), 101ull, term_ctr(CID_SCon, STAT_OFF + 1573), 114ull, term_ctr(CID_SCon, STAT_OFF + 1575), 32ull, term_ctr(CID_SCon, STAT_OFF + 1577), 115ull, term_ctr(CID_SCon, STAT_OFF + 1579), 114ull, term_ctr(CID_SCon, STAT_OFF + 1581), 111ull, term_ctr(CID_SCon, STAT_OFF + 1583), 116ull, term_ctr(CID_SCon, STAT_OFF + 1585), 97ull, term_ctr(CID_SCon, STAT_OFF + 1587), 114ull, term_ctr(CID_SCon, STAT_OFF + 1589), 101ull, term_ctr(CID_SCon, STAT_OFF + 1591), 112ull, term_ctr(CID_SCon, STAT_OFF + 1593), 111ull, term_ctr(CID_SCon, STAT_OFF + 1595), 32ull, term_ctr(CID_SCon, STAT_OFF + 1597), 116ull, term_ctr(CID_SCon, STAT_OFF + 1599), 101ull, term_ctr(CID_SCon, STAT_OFF + 1601), 115ull, term_ctr(CID_SCon, STAT_OFF + 1603), term_ctr(CID_SCon, STAT_OFF + 1605), 101ull, term_ctr(CID_SCon, STAT_OFF + 229), 115ull, term_ctr(CID_SCon, STAT_OFF + 1608), 32ull, term_ctr(CID_SCon, STAT_OFF + 1610), 101ull, term_ctr(CID_SCon, STAT_OFF + 1612), 116ull, term_ctr(CID_SCon, STAT_OFF + 1614), 105ull, term_ctr(CID_SCon, STAT_OFF + 1616), 110ull, term_ctr(CID_SCon, STAT_OFF + 1618), 105ull, term_ctr(CID_SCon, STAT_OFF + 1620), 102ull, term_ctr(CID_SCon, STAT_OFF + 1622), 32ull, term_ctr(CID_SCon, STAT_OFF + 1624), 97ull, term_ctr(CID_SCon, STAT_OFF + 1626), 32ull, term_ctr(CID_SCon, STAT_OFF + 1628), 115ull, term_ctr(CID_SCon, STAT_OFF + 1630), 101ull, term_ctr(CID_SCon, STAT_OFF + 1632), 114ull, term_ctr(CID_SCon, STAT_OFF + 1634), 105ull, term_ctr(CID_SCon, STAT_OFF + 1636), 117ull, term_ctr(CID_SCon, STAT_OFF + 1638), 113ull, term_ctr(CID_SCon, STAT_OFF + 1640), 101ull, term_ctr(CID_SCon, STAT_OFF + 1642), 114ull, term_ctr(CID_SCon, STAT_OFF + 1644), 32ull, term_ctr(CID_SCon, STAT_OFF + 1646), 112ull, term_ctr(CID_SCon, STAT_OFF + 1648), 105ull, term_ctr(CID_SCon, STAT_OFF + 1650), 104ull, term_ctr(CID_SCon, STAT_OFF + 1652), 115ull, term_ctr(CID_SCon, STAT_OFF + 1654), 114ull, term_ctr(CID_SCon, STAT_OFF + 1656), 101ull, term_ctr(CID_SCon, STAT_OFF + 1658), 98ull, term_ctr(CID_SCon, STAT_OFF + 1660), 109ull, term_ctr(CID_SCon, STAT_OFF + 1662), 101ull, term_ctr(CID_SCon, STAT_OFF + 1664), 109ull, term_ctr(CID_SCon, STAT_OFF + 1666), term_ctr(CID_SCon, STAT_OFF + 1668), 108ull, term_pak(CID_SNil, 0), 97ull, term_ctr(CID_SCon, STAT_OFF + 1671), 99ull, term_ctr(CID_SCon, STAT_OFF + 1673), 111ull, term_ctr(CID_SCon, STAT_OFF + 1675), 108ull, term_ctr(CID_SCon, STAT_OFF + 1677), 32ull, term_ctr(CID_SCon, STAT_OFF + 1679), 82ull, term_ctr(CID_SCon, STAT_OFF + 1681), 73ull, term_ctr(CID_SCon, STAT_OFF + 1683), 32ull, term_ctr(CID_SCon, STAT_OFF + 1685), 99ull, term_ctr(CID_SCon, STAT_OFF + 1687), 105ull, term_ctr(CID_SCon, STAT_OFF + 1689), 114ull, term_ctr(CID_SCon, STAT_OFF + 1691), 101ull, term_ctr(CID_SCon, STAT_OFF + 1693), 109ull, term_ctr(CID_SCon, STAT_OFF + 1695), 117ull, term_ctr(CID_SCon, STAT_OFF + 1697), 110ull, term_ctr(CID_SCon, STAT_OFF + 1699), 32ull, term_ctr(CID_SCon, STAT_OFF + 1701), 100ull, term_ctr(CID_SCon, STAT_OFF + 1703), 110ull, term_ctr(CID_SCon, STAT_OFF + 1705), 117ull, term_ctr(CID_SCon, STAT_OFF + 1707), 111ull, term_ctr(CID_SCon, STAT_OFF + 1709), 98ull, term_ctr(CID_SCon, STAT_OFF + 1711), 110ull, term_ctr(CID_SCon, STAT_OFF + 1713), 117ull, term_ctr(CID_SCon, STAT_OFF + 1715), term_ctr(CID_SCon, STAT_OFF + 1717), 105ull, term_ctr(CID_SCon, STAT_OFF + 168), 100ull, term_ctr(CID_SCon, STAT_OFF + 1720), 110ull, term_ctr(CID_SCon, STAT_OFF + 1722), 111ull, term_ctr(CID_SCon, STAT_OFF + 1724), 99ull, term_ctr(CID_SCon, STAT_OFF + 1726), 32ull, term_ctr(CID_SCon, STAT_OFF + 1728), 110ull, term_ctr(CID_SCon, STAT_OFF + 1730), 97ull, term_ctr(CID_SCon, STAT_OFF + 1732), 101ull, term_ctr(CID_SCon, STAT_OFF + 1734), 108ull, term_ctr(CID_SCon, STAT_OFF + 1736), 111ull, term_ctr(CID_SCon, STAT_OFF + 1738), 111ull, term_ctr(CID_SCon, STAT_OFF + 1740), 98ull, term_ctr(CID_SCon, STAT_OFF + 1742), 32ull, term_ctr(CID_SCon, STAT_OFF + 1744), 97ull, term_ctr(CID_SCon, STAT_OFF + 1746), 32ull, term_ctr(CID_SCon, STAT_OFF + 1748), 100ull, term_ctr(CID_SCon, STAT_OFF + 1750), 101ull, term_ctr(CID_SCon, STAT_OFF + 1752), 116ull, term_ctr(CID_SCon, STAT_OFF + 1754), 99ull, term_ctr(CID_SCon, STAT_OFF + 1756), 101ull, term_ctr(CID_SCon, STAT_OFF + 1758), 112ull, term_ctr(CID_SCon, STAT_OFF + 1760), 120ull, term_ctr(CID_SCon, STAT_OFF + 1762), 101ull, term_ctr(CID_SCon, STAT_OFF + 1764), term_ctr(CID_SCon, STAT_OFF + 1766), 97ull, term_ctr(CID_SCon, STAT_OFF + 168), 99ull, term_ctr(CID_SCon, STAT_OFF + 1769), 105ull, term_ctr(CID_SCon, STAT_OFF + 1771), 108ull, term_ctr(CID_SCon, STAT_OFF + 1773), 112ull, term_ctr(CID_SCon, STAT_OFF + 1775), 109ull, term_ctr(CID_SCon, STAT_OFF + 1777), 105ull, term_ctr(CID_SCon, STAT_OFF + 1779), 32ull, term_ctr(CID_SCon, STAT_OFF + 1781), 114ull, term_ctr(CID_SCon, STAT_OFF + 1783), 111ull, term_ctr(CID_SCon, STAT_OFF + 1785), 102ull, term_ctr(CID_SCon, STAT_OFF + 1787), 32ull, term_ctr(CID_SCon, STAT_OFF + 1789), 115ull, term_ctr(CID_SCon, STAT_OFF + 1791), 110ull, term_ctr(CID_SCon, STAT_OFF + 1793), 97ull, term_ctr(CID_SCon, STAT_OFF + 1795), 101ull, term_ctr(CID_SCon, STAT_OFF + 1797), 108ull, term_ctr(CID_SCon, STAT_OFF + 1799), 111ull, term_ctr(CID_SCon, STAT_OFF + 1801), 111ull, term_ctr(CID_SCon, STAT_OFF + 1803), 98ull, term_ctr(CID_SCon, STAT_OFF + 1805), 32ull, term_ctr(CID_SCon, STAT_OFF + 1807), 100ull, term_ctr(CID_SCon, STAT_OFF + 1809), 101ull, term_ctr(CID_SCon, STAT_OFF + 1811), 116ull, term_ctr(CID_SCon, STAT_OFF + 1813), 99ull, term_ctr(CID_SCon, STAT_OFF + 1815), 101ull, term_ctr(CID_SCon, STAT_OFF + 1817), 112ull, term_ctr(CID_SCon, STAT_OFF + 1819), 120ull, term_ctr(CID_SCon, STAT_OFF + 1821), 101ull, term_ctr(CID_SCon, STAT_OFF + 1823), term_ctr(CID_SCon, STAT_OFF + 1825), 126ull, term_pak(CID_SNil, 0), 32ull, term_ctr(CID_SCon, STAT_OFF + 1828), 114ull, term_ctr(CID_SCon, STAT_OFF + 1830), 111ull, term_ctr(CID_SCon, STAT_OFF + 1832), 102ull, term_ctr(CID_SCon, STAT_OFF + 1834), 32ull, term_ctr(CID_SCon, STAT_OFF + 1836), 110ull, term_ctr(CID_SCon, STAT_OFF + 1838), 97ull, term_ctr(CID_SCon, STAT_OFF + 1840), 101ull, term_ctr(CID_SCon, STAT_OFF + 1842), 108ull, term_ctr(CID_SCon, STAT_OFF + 1844), 111ull, term_ctr(CID_SCon, STAT_OFF + 1846), 111ull, term_ctr(CID_SCon, STAT_OFF + 1848), 98ull, term_ctr(CID_SCon, STAT_OFF + 1850), 32ull, term_ctr(CID_SCon, STAT_OFF + 1852), 97ull, term_ctr(CID_SCon, STAT_OFF + 1854), 32ull, term_ctr(CID_SCon, STAT_OFF + 1856), 100ull, term_ctr(CID_SCon, STAT_OFF + 1858), 101ull, term_ctr(CID_SCon, STAT_OFF + 1860), 116ull, term_ctr(CID_SCon, STAT_OFF + 1862), 99ull, term_ctr(CID_SCon, STAT_OFF + 1864), 101ull, term_ctr(CID_SCon, STAT_OFF + 1866), 112ull, term_ctr(CID_SCon, STAT_OFF + 1868), 120ull, term_ctr(CID_SCon, STAT_OFF + 1870), 101ull, term_ctr(CID_SCon, STAT_OFF + 1872), term_ctr(CID_SCon, STAT_OFF + 1874), 47ull, term_pak(CID_SNil, 0), 92ull, term_ctr(CID_SCon, STAT_OFF + 1877), 32ull, term_ctr(CID_SCon, STAT_OFF + 1879), 114ull, term_ctr(CID_SCon, STAT_OFF + 1881), 111ull, term_ctr(CID_SCon, STAT_OFF + 1883), 102ull, term_ctr(CID_SCon, STAT_OFF + 1885), 32ull, term_ctr(CID_SCon, STAT_OFF + 1887), 115ull, term_ctr(CID_SCon, STAT_OFF + 1889), 110ull, term_ctr(CID_SCon, STAT_OFF + 1891), 97ull, term_ctr(CID_SCon, STAT_OFF + 1893), 101ull, term_ctr(CID_SCon, STAT_OFF + 1895), 108ull, term_ctr(CID_SCon, STAT_OFF + 1897), 111ull, term_ctr(CID_SCon, STAT_OFF + 1899), 111ull, term_ctr(CID_SCon, STAT_OFF + 1901), 98ull, term_ctr(CID_SCon, STAT_OFF + 1903), 32ull, term_ctr(CID_SCon, STAT_OFF + 1905), 100ull, term_ctr(CID_SCon, STAT_OFF + 1907), 101ull, term_ctr(CID_SCon, STAT_OFF + 1909), 116ull, term_ctr(CID_SCon, STAT_OFF + 1911), 99ull, term_ctr(CID_SCon, STAT_OFF + 1913), 101ull, term_ctr(CID_SCon, STAT_OFF + 1915), 112ull, term_ctr(CID_SCon, STAT_OFF + 1917), 120ull, term_ctr(CID_SCon, STAT_OFF + 1919), 101ull, term_ctr(CID_SCon, STAT_OFF + 1921), term_ctr(CID_SCon, STAT_OFF + 1923), 92ull, term_pak(CID_SNil, 0), 47ull, term_ctr(CID_SCon, STAT_OFF + 1926), 32ull, term_ctr(CID_SCon, STAT_OFF + 1928), 114ull, term_ctr(CID_SCon, STAT_OFF + 1930), 111ull, term_ctr(CID_SCon, STAT_OFF + 1932), 102ull, term_ctr(CID_SCon, STAT_OFF + 1934), 32ull, term_ctr(CID_SCon, STAT_OFF + 1936), 115ull, term_ctr(CID_SCon, STAT_OFF + 1938), 110ull, term_ctr(CID_SCon, STAT_OFF + 1940), 97ull, term_ctr(CID_SCon, STAT_OFF + 1942), 101ull, term_ctr(CID_SCon, STAT_OFF + 1944), 108ull, term_ctr(CID_SCon, STAT_OFF + 1946), 111ull, term_ctr(CID_SCon, STAT_OFF + 1948), 111ull, term_ctr(CID_SCon, STAT_OFF + 1950), 98ull, term_ctr(CID_SCon, STAT_OFF + 1952), 32ull, term_ctr(CID_SCon, STAT_OFF + 1954), 100ull, term_ctr(CID_SCon, STAT_OFF + 1956), 101ull, term_ctr(CID_SCon, STAT_OFF + 1958), 116ull, term_ctr(CID_SCon, STAT_OFF + 1960), 99ull, term_ctr(CID_SCon, STAT_OFF + 1962), 101ull, term_ctr(CID_SCon, STAT_OFF + 1964), 112ull, term_ctr(CID_SCon, STAT_OFF + 1966), 120ull, term_ctr(CID_SCon, STAT_OFF + 1968), 101ull, term_ctr(CID_SCon, STAT_OFF + 1970), term_ctr(CID_SCon, STAT_OFF + 1972), 61ull, term_pak(CID_SNil, 0), 60ull, term_ctr(CID_SCon, STAT_OFF + 1975), 32ull, term_ctr(CID_SCon, STAT_OFF + 1977), 114ull, term_ctr(CID_SCon, STAT_OFF + 1979), 111ull, term_ctr(CID_SCon, STAT_OFF + 1981), 102ull, term_ctr(CID_SCon, STAT_OFF + 1983), 32ull, term_ctr(CID_SCon, STAT_OFF + 1985), 115ull, term_ctr(CID_SCon, STAT_OFF + 1987), 114ull, term_ctr(CID_SCon, STAT_OFF + 1989), 101ull, term_ctr(CID_SCon, STAT_OFF + 1991), 103ull, term_ctr(CID_SCon, STAT_OFF + 1993), 101ull, term_ctr(CID_SCon, STAT_OFF + 1995), 116ull, term_ctr(CID_SCon, STAT_OFF + 1997), 110ull, term_ctr(CID_SCon, STAT_OFF + 1999), 105ull, term_ctr(CID_SCon, STAT_OFF + 2001), 32ull, term_ctr(CID_SCon, STAT_OFF + 2003), 100ull, term_ctr(CID_SCon, STAT_OFF + 2005), 101ull, term_ctr(CID_SCon, STAT_OFF + 2007), 116ull, term_ctr(CID_SCon, STAT_OFF + 2009), 99ull, term_ctr(CID_SCon, STAT_OFF + 2011), 101ull, term_ctr(CID_SCon, STAT_OFF + 2013), 112ull, term_ctr(CID_SCon, STAT_OFF + 2015), 120ull, term_ctr(CID_SCon, STAT_OFF + 2017), 101ull, term_ctr(CID_SCon, STAT_OFF + 2019), term_ctr(CID_SCon, STAT_OFF + 2021), 60ull, term_pak(CID_SNil, 0), 32ull, term_ctr(CID_SCon, STAT_OFF + 2024), 114ull, term_ctr(CID_SCon, STAT_OFF + 2026), 111ull, term_ctr(CID_SCon, STAT_OFF + 2028), 102ull, term_ctr(CID_SCon, STAT_OFF + 2030), 32ull, term_ctr(CID_SCon, STAT_OFF + 2032), 115ull, term_ctr(CID_SCon, STAT_OFF + 2034), 114ull, term_ctr(CID_SCon, STAT_OFF + 2036), 101ull, term_ctr(CID_SCon, STAT_OFF + 2038), 103ull, term_ctr(CID_SCon, STAT_OFF + 2040), 101ull, term_ctr(CID_SCon, STAT_OFF + 2042), 116ull, term_ctr(CID_SCon, STAT_OFF + 2044), 110ull, term_ctr(CID_SCon, STAT_OFF + 2046), 105ull, term_ctr(CID_SCon, STAT_OFF + 2048), 32ull, term_ctr(CID_SCon, STAT_OFF + 2050), 100ull, term_ctr(CID_SCon, STAT_OFF + 2052), 101ull, term_ctr(CID_SCon, STAT_OFF + 2054), 116ull, term_ctr(CID_SCon, STAT_OFF + 2056), 99ull, term_ctr(CID_SCon, STAT_OFF + 2058), 101ull, term_ctr(CID_SCon, STAT_OFF + 2060), 112ull, term_ctr(CID_SCon, STAT_OFF + 2062), 120ull, term_ctr(CID_SCon, STAT_OFF + 2064), 101ull, term_ctr(CID_SCon, STAT_OFF + 2066), term_ctr(CID_SCon, STAT_OFF + 2068), 45ull, term_pak(CID_SNil, 0), 32ull, term_ctr(CID_SCon, STAT_OFF + 2071), 121ull, term_ctr(CID_SCon, STAT_OFF + 2073), 114ull, term_ctr(CID_SCon, STAT_OFF + 2075), 97ull, term_ctr(CID_SCon, STAT_OFF + 2077), 110ull, term_ctr(CID_SCon, STAT_OFF + 2079), 117ull, term_ctr(CID_SCon, STAT_OFF + 2081), 32ull, term_ctr(CID_SCon, STAT_OFF + 2083), 114ull, term_ctr(CID_SCon, STAT_OFF + 2085), 111ull, term_ctr(CID_SCon, STAT_OFF + 2087), 102ull, term_ctr(CID_SCon, STAT_OFF + 2089), 32ull, term_ctr(CID_SCon, STAT_OFF + 2091), 114ull, term_ctr(CID_SCon, STAT_OFF + 2093), 101ull, term_ctr(CID_SCon, STAT_OFF + 2095), 103ull, term_ctr(CID_SCon, STAT_OFF + 2097), 101ull, term_ctr(CID_SCon, STAT_OFF + 2099), 116ull, term_ctr(CID_SCon, STAT_OFF + 2101), 110ull, term_ctr(CID_SCon, STAT_OFF + 2103), 105ull, term_ctr(CID_SCon, STAT_OFF + 2105), 32ull, term_ctr(CID_SCon, STAT_OFF + 2107), 110ull, term_ctr(CID_SCon, STAT_OFF + 2109), 97ull, term_ctr(CID_SCon, STAT_OFF + 2111), 32ull, term_ctr(CID_SCon, STAT_OFF + 2113), 100ull, term_ctr(CID_SCon, STAT_OFF + 2115), 101ull, term_ctr(CID_SCon, STAT_OFF + 2117), 116ull, term_ctr(CID_SCon, STAT_OFF + 2119), 99ull, term_ctr(CID_SCon, STAT_OFF + 2121), 101ull, term_ctr(CID_SCon, STAT_OFF + 2123), 112ull, term_ctr(CID_SCon, STAT_OFF + 2125), 120ull, term_ctr(CID_SCon, STAT_OFF + 2127), 101ull, term_ctr(CID_SCon, STAT_OFF + 2129), term_ctr(CID_SCon, STAT_OFF + 2131), 114ull, term_ctr(CID_SCon, STAT_OFF + 2073), 111ull, term_ctr(CID_SCon, STAT_OFF + 2134), 102ull, term_ctr(CID_SCon, STAT_OFF + 2136), 32ull, term_ctr(CID_SCon, STAT_OFF + 2138), 115ull, term_ctr(CID_SCon, STAT_OFF + 2140), 114ull, term_ctr(CID_SCon, STAT_OFF + 2142), 101ull, term_ctr(CID_SCon, STAT_OFF + 2144), 103ull, term_ctr(CID_SCon, STAT_OFF + 2146), 101ull, term_ctr(CID_SCon, STAT_OFF + 2148), 116ull, term_ctr(CID_SCon, STAT_OFF + 2150), 110ull, term_ctr(CID_SCon, STAT_OFF + 2152), 105ull, term_ctr(CID_SCon, STAT_OFF + 2154), 32ull, term_ctr(CID_SCon, STAT_OFF + 2156), 100ull, term_ctr(CID_SCon, STAT_OFF + 2158), 101ull, term_ctr(CID_SCon, STAT_OFF + 2160), 116ull, term_ctr(CID_SCon, STAT_OFF + 2162), 99ull, term_ctr(CID_SCon, STAT_OFF + 2164), 101ull, term_ctr(CID_SCon, STAT_OFF + 2166), 112ull, term_ctr(CID_SCon, STAT_OFF + 2168), 120ull, term_ctr(CID_SCon, STAT_OFF + 2170), 101ull, term_ctr(CID_SCon, STAT_OFF + 2172), term_ctr(CID_SCon, STAT_OFF + 2174), 43ull, term_pak(CID_SNil, 0), 32ull, term_ctr(CID_SCon, STAT_OFF + 2177), 114ull, term_ctr(CID_SCon, STAT_OFF + 2179), 111ull, term_ctr(CID_SCon, STAT_OFF + 2181), 102ull, term_ctr(CID_SCon, STAT_OFF + 2183), 32ull, term_ctr(CID_SCon, STAT_OFF + 2185), 115ull, term_ctr(CID_SCon, STAT_OFF + 2187), 114ull, term_ctr(CID_SCon, STAT_OFF + 2189), 101ull, term_ctr(CID_SCon, STAT_OFF + 2191), 103ull, term_ctr(CID_SCon, STAT_OFF + 2193), 101ull, term_ctr(CID_SCon, STAT_OFF + 2195), 116ull, term_ctr(CID_SCon, STAT_OFF + 2197), 110ull, term_ctr(CID_SCon, STAT_OFF + 2199), 105ull, term_ctr(CID_SCon, STAT_OFF + 2201), 32ull, term_ctr(CID_SCon, STAT_OFF + 2203), 100ull, term_ctr(CID_SCon, STAT_OFF + 2205), 101ull, term_ctr(CID_SCon, STAT_OFF + 2207), 116ull, term_ctr(CID_SCon, STAT_OFF + 2209), 99ull, term_ctr(CID_SCon, STAT_OFF + 2211), 101ull, term_ctr(CID_SCon, STAT_OFF + 2213), 112ull, term_ctr(CID_SCon, STAT_OFF + 2215), 120ull, term_ctr(CID_SCon, STAT_OFF + 2217), 101ull, term_ctr(CID_SCon, STAT_OFF + 2219), term_ctr(CID_SCon, STAT_OFF + 2221), 97ull, term_ctr(CID_SCon, STAT_OFF + 192), 32ull, term_ctr(CID_SCon, STAT_OFF + 2224), 115ull, term_ctr(CID_SCon, STAT_OFF + 2226), 101ull, term_ctr(CID_SCon, STAT_OFF + 2228), 114ull, term_ctr(CID_SCon, STAT_OFF + 2230), 105ull, term_ctr(CID_SCon, STAT_OFF + 2232), 117ull, term_ctr(CID_SCon, STAT_OFF + 2234), 113ull, term_ctr(CID_SCon, STAT_OFF + 2236), 101ull, term_ctr(CID_SCon, STAT_OFF + 2238), 114ull, term_ctr(CID_SCon, STAT_OFF + 2240), 32ull, term_ctr(CID_SCon, STAT_OFF + 2242), 78ull, term_ctr(CID_SCon, STAT_OFF + 2244), 73ull, term_ctr(CID_SCon, STAT_OFF + 2246), 65ull, term_ctr(CID_SCon, STAT_OFF + 2248), 77ull, term_ctr(CID_SCon, STAT_OFF + 2250), 79ull, term_ctr(CID_SCon, STAT_OFF + 2252), 68ull, term_ctr(CID_SCon, STAT_OFF + 2254), term_ctr(CID_SCon, STAT_OFF + 2256), 84ull, term_ctr(CID_SCon, STAT_OFF + 2244), 80ull, term_ctr(CID_SCon, STAT_OFF + 2259), 69ull, term_ctr(CID_SCon, STAT_OFF + 2261), 67ull, term_ctr(CID_SCon, STAT_OFF + 2263), 88ull, term_ctr(CID_SCon, STAT_OFF + 2265), 69ull, term_ctr(CID_SCon, STAT_OFF + 2267), term_ctr(CID_SCon, STAT_OFF + 2269), 112ull, term_ctr(CID_SCon, STAT_OFF + 2244), 117ull, term_ctr(CID_SCon, STAT_OFF + 2272), 107ull, term_ctr(CID_SCon, STAT_OFF + 2274), 111ull, term_ctr(CID_SCon, STAT_OFF + 2276), 111ull, term_ctr(CID_SCon, STAT_OFF + 2278), 108ull, term_ctr(CID_SCon, STAT_OFF + 2280), term_ctr(CID_SCon, STAT_OFF + 2282), 110ull, term_ctr(CID_SCon, STAT_OFF + 1648), 111ull, term_ctr(CID_SCon, STAT_OFF + 2285), 105ull, term_ctr(CID_SCon, STAT_OFF + 2287), 115ull, term_ctr(CID_SCon, STAT_OFF + 2289), 115ull, term_ctr(CID_SCon, STAT_OFF + 2291), 101ull, term_ctr(CID_SCon, STAT_OFF + 2293), 114ull, term_ctr(CID_SCon, STAT_OFF + 2295), 112ull, term_ctr(CID_SCon, STAT_OFF + 2297), 120ull, term_ctr(CID_SCon, STAT_OFF + 2299), 101ull, term_ctr(CID_SCon, STAT_OFF + 2301), 32ull, term_ctr(CID_SCon, STAT_OFF + 2303), 100ull, term_ctr(CID_SCon, STAT_OFF + 2305), 101ull, term_ctr(CID_SCon, STAT_OFF + 2307), 100ull, term_ctr(CID_SCon, STAT_OFF + 2309), 110ull, term_ctr(CID_SCon, STAT_OFF + 2311), 117ull, term_ctr(CID_SCon, STAT_OFF + 2313), 111ull, term_ctr(CID_SCon, STAT_OFF + 2315), 98ull, term_ctr(CID_SCon, STAT_OFF + 2317), term_ctr(CID_SCon, STAT_OFF + 2319), 101ull, term_ctr(CID_SCon, STAT_OFF + 1038), 114ull, term_ctr(CID_SCon, STAT_OFF + 2322), 105ull, term_ctr(CID_SCon, STAT_OFF + 2324), 117ull, term_ctr(CID_SCon, STAT_OFF + 2326), 113ull, term_ctr(CID_SCon, STAT_OFF + 2328), 101ull, term_ctr(CID_SCon, STAT_OFF + 2330), 114ull, term_ctr(CID_SCon, STAT_OFF + 2332), 32ull, term_ctr(CID_SCon, STAT_OFF + 2334), 100ull, term_ctr(CID_SCon, STAT_OFF + 2336), 110ull, term_ctr(CID_SCon, STAT_OFF + 2338), 97ull, term_ctr(CID_SCon, STAT_OFF + 2340), 114ull, term_ctr(CID_SCon, STAT_OFF + 2342), 101ull, term_ctr(CID_SCon, STAT_OFF + 2344), 112ull, term_ctr(CID_SCon, STAT_OFF + 2346), 111ull, term_ctr(CID_SCon, STAT_OFF + 2348), 32ull, term_ctr(CID_SCon, STAT_OFF + 2350), 116ull, term_ctr(CID_SCon, STAT_OFF + 2352), 101ull, term_ctr(CID_SCon, STAT_OFF + 2354), 115ull, term_ctr(CID_SCon, STAT_OFF + 2356), 32ull, term_ctr(CID_SCon, STAT_OFF + 2358), 101ull, term_ctr(CID_SCon, STAT_OFF + 2360), 116ull, term_ctr(CID_SCon, STAT_OFF + 2362), 105ull, term_ctr(CID_SCon, STAT_OFF + 2364), 110ull, term_ctr(CID_SCon, STAT_OFF + 2366), 105ull, term_ctr(CID_SCon, STAT_OFF + 2368), 102ull, term_ctr(CID_SCon, STAT_OFF + 2370), term_ctr(CID_SCon, STAT_OFF + 2372), 98ull, term_ctr(CID_SCon, STAT_OFF + 670), 97ull, term_ctr(CID_SCon, STAT_OFF + 2375), 108ull, term_ctr(CID_SCon, STAT_OFF + 2377), 105ull, term_ctr(CID_SCon, STAT_OFF + 2379), 97ull, term_ctr(CID_SCon, STAT_OFF + 2381), 118ull, term_ctr(CID_SCon, STAT_OFF + 2383), 97ull, term_ctr(CID_SCon, STAT_OFF + 2385), 110ull, term_ctr(CID_SCon, STAT_OFF + 2387), 117ull, term_ctr(CID_SCon, STAT_OFF + 2389), 32ull, term_ctr(CID_SCon, STAT_OFF + 2391), 115ull, term_ctr(CID_SCon, STAT_OFF + 2393), 105ull, term_ctr(CID_SCon, STAT_OFF + 2395), 32ull, term_ctr(CID_SCon, STAT_OFF + 2397), 32ull, term_pak(CID_SNil, 0), 116ull, term_ctr(CID_SCon, STAT_OFF + 2401), 111ull, term_ctr(CID_SCon, STAT_OFF + 2403), 108ull, term_ctr(CID_SCon, STAT_OFF + 2405), 115ull, term_ctr(CID_SCon, STAT_OFF + 2407), 32ull, term_ctr(CID_SCon, STAT_OFF + 2409), 101ull, term_ctr(CID_SCon, STAT_OFF + 2411), 116ull, term_ctr(CID_SCon, STAT_OFF + 2413), 97ull, term_ctr(CID_SCon, STAT_OFF + 2415), 116ull, term_ctr(CID_SCon, STAT_OFF + 2417), 115ull, term_ctr(CID_SCon, STAT_OFF + 2419), 115ull, term_ctr(CID_SCon, STAT_OFF + 166), 115ull, term_ctr(CID_SCon, STAT_OFF + 2423), 101ull, term_ctr(CID_SCon, STAT_OFF + 2425), 114ull, term_ctr(CID_SCon, STAT_OFF + 2427), 112ull, term_ctr(CID_SCon, STAT_OFF + 2429), 120ull, term_ctr(CID_SCon, STAT_OFF + 2431), 101ull, term_ctr(CID_SCon, STAT_OFF + 2433), 32ull, term_ctr(CID_SCon, STAT_OFF + 2435), 82ull, term_ctr(CID_SCon, STAT_OFF + 2437), 73ull, term_ctr(CID_SCon, STAT_OFF + 2439), 32ull, term_ctr(CID_SCon, STAT_OFF + 2441), 103ull, term_ctr(CID_SCon, STAT_OFF + 2443), 110ull, term_ctr(CID_SCon, STAT_OFF + 2445), 105ull, term_ctr(CID_SCon, STAT_OFF + 2447), 115ull, term_ctr(CID_SCon, STAT_OFF + 2449), 115ull, term_ctr(CID_SCon, STAT_OFF + 2451), 105ull, term_ctr(CID_SCon, STAT_OFF + 2453), 109ull, term_ctr(CID_SCon, STAT_OFF + 2455), 48ull, term_pak(CID_SNil, 0), 0ull, 1ull, term_ctr(CID_ir_IAdd, STAT_OFF + 2461), term_pak(CID_Nil, 0), term_pak(CID_ir_IInt, 1ull), term_ctr(CID_Con, STAT_OFF + 2463), term_pak(CID_ir_ISlot, 0ull), term_ctr(CID_Con, STAT_OFF + 2465), term_pak(CID_ir_VI, 2ull), term_pak(CID_Nil, 0), 125ull, term_pak(CID_SNil, 0), 58ull, term_pak(CID_SNil, 0), 34ull, term_ctr(CID_SCon, STAT_OFF + 2473), 101ull, term_ctr(CID_SCon, STAT_OFF + 2475), 117ull, term_ctr(CID_SCon, STAT_OFF + 2477), 108ull, term_ctr(CID_SCon, STAT_OFF + 2479), 97ull, term_ctr(CID_SCon, STAT_OFF + 2481), 118ull, term_ctr(CID_SCon, STAT_OFF + 2483), 34ull, term_ctr(CID_SCon, STAT_OFF + 2485), 44ull, term_ctr(CID_SCon, STAT_OFF + 2487), 112ull, term_ctr(CID_SCon, STAT_OFF + 2475), 97ull, term_ctr(CID_SCon, STAT_OFF + 2491), 101ull, term_ctr(CID_SCon, STAT_OFF + 2493), 104ull, term_ctr(CID_SCon, STAT_OFF + 2495), 34ull, term_ctr(CID_SCon, STAT_OFF + 2497), 44ull, term_ctr(CID_SCon, STAT_OFF + 2499), 115ull, term_ctr(CID_SCon, STAT_OFF + 2475), 116ull, term_ctr(CID_SCon, STAT_OFF + 2503), 110ull, term_ctr(CID_SCon, STAT_OFF + 2505), 101ull, term_ctr(CID_SCon, STAT_OFF + 2507), 118ull, term_ctr(CID_SCon, STAT_OFF + 2509), 101ull, term_ctr(CID_SCon, STAT_OFF + 2511), 34ull, term_ctr(CID_SCon, STAT_OFF + 2513), 44ull, term_ctr(CID_SCon, STAT_OFF + 2515), 100ull, term_ctr(CID_SCon, STAT_OFF + 2475), 110ull, term_ctr(CID_SCon, STAT_OFF + 2519), 101ull, term_ctr(CID_SCon, STAT_OFF + 2521), 107ull, term_ctr(CID_SCon, STAT_OFF + 2523), 99ull, term_ctr(CID_SCon, STAT_OFF + 2525), 97ull, term_ctr(CID_SCon, STAT_OFF + 2527), 98ull, term_ctr(CID_SCon, STAT_OFF + 2529), 34ull, term_ctr(CID_SCon, STAT_OFF + 2531), 123ull, term_ctr(CID_SCon, STAT_OFF + 2533), 82ull, term_ctr(CID_SCon, STAT_OFF + 467), 73ull, term_ctr(CID_SCon, STAT_OFF + 2537), 32ull, term_ctr(CID_SCon, STAT_OFF + 2539), 100ull, term_ctr(CID_SCon, STAT_OFF + 2541), 101ull, term_ctr(CID_SCon, STAT_OFF + 2543), 116ull, term_ctr(CID_SCon, STAT_OFF + 2545), 99ull, term_ctr(CID_SCon, STAT_OFF + 2547), 101ull, term_ctr(CID_SCon, STAT_OFF + 2549), 112ull, term_ctr(CID_SCon, STAT_OFF + 2551), 120ull, term_ctr(CID_SCon, STAT_OFF + 2553), 101ull, term_ctr(CID_SCon, STAT_OFF + 2555), 110ull, term_ctr(CID_SCon, STAT_OFF + 2557), 117ull, term_ctr(CID_SCon, STAT_OFF + 2559), 120ull, term_pak(CID_SNil, 0), 101ull, term_ctr(CID_SCon, STAT_OFF + 2563), 100ull, term_ctr(CID_SCon, STAT_OFF + 2565), 110ull, term_ctr(CID_SCon, STAT_OFF + 2567), 105ull, term_ctr(CID_SCon, STAT_OFF + 2569), 32ull, term_ctr(CID_SCon, STAT_OFF + 2571), 101ull, term_ctr(CID_SCon, STAT_OFF + 2573), 100ull, term_ctr(CID_SCon, STAT_OFF + 2575), 111ull, term_ctr(CID_SCon, STAT_OFF + 2577), 99ull, term_ctr(CID_SCon, STAT_OFF + 2579), 112ull, term_ctr(CID_SCon, STAT_OFF + 2581), 111ull, term_ctr(CID_SCon, STAT_OFF + 2583), 32ull, term_ctr(CID_SCon, STAT_OFF + 2585), 100ull, term_ctr(CID_SCon, STAT_OFF + 2587), 101ull, term_ctr(CID_SCon, STAT_OFF + 2589), 116ull, term_ctr(CID_SCon, STAT_OFF + 2591), 99ull, term_ctr(CID_SCon, STAT_OFF + 2593), 101ull, term_ctr(CID_SCon, STAT_OFF + 2595), 112ull, term_ctr(CID_SCon, STAT_OFF + 2597), 120ull, term_ctr(CID_SCon, STAT_OFF + 2599), 101ull, term_ctr(CID_SCon, STAT_OFF + 2601) };

INLINE Term spin_ir_functionNode(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3);
INLINE Term spin_ir_sequenceDrop(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_functionStack(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_functionHeight(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_functionAppendBalance(Env e, THR Term* o, u32 r0, Term r1, Term r2, Term r3, Term r4);
INLINE Term spin_ir_sle(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_functionSize(Env e, THR Term* o, Term r0);
INLINE Term spin_Bool_and(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_buildFunctionRight(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3, Term r4, Term r5);
INLINE Term spin_List_reverse(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_functionAppendBalanced(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3);
INLINE Term spin_ir_slt(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_emptyFunction(Env e, THR Term* o);
INLINE Term spin_ir_sge(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_domainTreeRight(Env e, THR Term* o, Term r0, Term r1, u32 r2, Term r3, Term r4);
INLINE Term spin_ir_functionMax(Env e, THR Term* o, u32 r0, u32 r1, u32 r2);
INLINE Term spin_ir_domainSize(Env e, THR Term* o, Term r0);
INLINE Term spin_List_reverse_go(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_List_merge_step(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3, Term r4, u32 r5);
INLINE Term spin_ir_sgt(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_domainBuilt(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_functionBuiltValue(Env e, THR Term* o, Term r0, Term r1, Term r2, u32 r3);
INLINE Term spin_Cmp_is_le(Env e, THR Term* o, u32 r0);
INLINE Term spin_ir_functionFound(Env e, THR Term* o, u32 r0, Term r1);
INLINE Term spin_String_order_go(Env e, THR Term* o, Term r0, Term r1, u32 r2);
INLINE Term spin_ir_negative(Env e, THR Term* o, u32 r0);
INLINE Term spin_ir_signedOrderSigns(Env e, THR Term* o, u32 r0, u32 r1, u32 r2, u32 r3);
INLINE Term spin_ir_domainTreeSize(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_domainRangeRank(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_List_sort_pair(Env e, THR Term* o, Term r0, Term r1, u32 r2);
INLINE Term spin_String_order(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_Cmp_is_eq(Env e, THR Term* o, u32 r0);
INLINE Term spin_ir_sequenceLengthArg(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_signedOrder(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_valueTag(Env e, THR Term* o, Term r0);
INLINE Term spin_Bool_to_u32(Env e, THR Term* o, u32 r0);
INLINE Term spin_ir_emptySet(Env e, THR Term* o);
INLINE Term spin_ir_domainRangeIndex(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_Bool_not(Env e, THR Term* o, u32 r0);
INLINE Term spin_ir_slt_go(Env e, THR Term* o, u32 r0, u32 r1, u32 r2, u32 r3);
INLINE Term spin_ir_boolEq(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_String_eq(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_functionEqual(Env e, THR Term* o, u32 r0);
INLINE Term spin_ir_pickVal(Env e, THR Term* o, u32 r0, Term r1, Term r2);
INLINE Term spin_ir_bindLocal(Env e, THR Term* o, Term r0, Term r1, u32 r2, Term r3);
INLINE Term spin_ir_productCountEnd(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_memberResult(Env e, THR Term* o, u32 r0, u32 r1, u32 r2);
INLINE Term spin_ir_booleanValue(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_vle(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_vlt(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_vnot(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_sneg(Env e, THR Term* o, u32 r0);
INLINE Term spin_ir_ssub(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_sadd(Env e, THR Term* o, u32 r0, u32 r1);
INLINE Term spin_ir_setInputErrorValue(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_nthVal(Env e, THR Term* o, Term r0, Term r1, Term r2);
INLINE Term spin_ir_vadd(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_vsub(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_vneg(Env e, THR Term* o, Term r0);
INLINE Term spin_ir_vgt(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_ir_vge(Env e, THR Term* o, Term r0, Term r1);
DFAR Term spin_ir_nthNode(Env e, THR Term* o, Term r0, Term r1);
INLINE Term spin_IO_die(Env e, THR Term* o, u32 r0, Term r1, Term r2);

INLINE Term spin_ir_functionNode(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3) {
  u32 wpoll = 0;
  Term _left_0 = r0;
  Term _key_0 = r1;
  Term _value_0 = r2;
  Term _right_0 = r3;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_functionHeight(e, _o_0, _left_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_ir_functionHeight(e, _o_1, _right_0) == 0) {
      return 0;
    }
    _left_0 = term_keep(e, _left_0, 1);
    _right_0 = term_keep(e, _right_0, 1);
    Term _o_2[1];
    if (spin_ir_functionSize(e, _o_2, _left_0) == 0) {
      return 0;
    }
    term_sink(e, _left_0);
    Term _o_3[1];
    if (spin_ir_functionSize(e, _o_3, _right_0) == 0) {
      return 0;
    }
    term_sink(e, _right_0);
    Term _a_0 = U32_BIN((u32)_o_2[0], +, (u32)_o_3[0]);
    Term _o_4[1];
    if (spin_ir_functionMax(e, _o_4, U32_BIN((u32)_o_0[0], >=, (u32)_o_1[0]), (u32)_o_0[0], (u32)_o_1[0]) == 0) {
      return 0;
    }
    u64 _nd_0 = heap_alloc(e, 3);
    e.mem[_nd_0 + 0] = rfc_seal(e, _left_0);
    e.mem[_nd_0 + 1] = rfc_seal(e, _key_0);
    e.mem[_nd_0 + 2] = rfc_seal(e, _value_0);
    e.mem[_nd_0 + 3] = rfc_seal(e, _right_0);
    e.mem[_nd_0 + 4] = rfc_seal(e, U32_BIN(_a_0, +, 1));
    e.mem[_nd_0 + 5] = rfc_seal(e, U32_BIN((u32)_o_4[0], +, 1));
    o[0] = term_ctr(CID_ir_FunctionNode, _nd_0);
  break;
  }
  return 1;
}

INLINE Term spin_ir_sequenceDrop(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _values_0 = r0;
  Term _count_0 = r1;
  WL_SPIN
    if (term_aux(_values_0) == CID_Con) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (_count_0 == 0) {
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, _f_1);
        o[0] = term_ctr(CID_Con, _nd_0);
      } else {
        term_sink(e, _f_0);
        Term _rest_0 = (_count_0 - 1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_1;
        r1 = _rest_0;
        _values_0 = r0;
        _count_0 = r1;
        WL_AGAIN(spin_ir_sequenceDrop);
      }
    } else {
      if (_count_0 == 0) {
        o[0] = term_pak(CID_Nil, 0);
      } else {
        o[0] = term_pak(CID_Nil, 0);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionStack(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _tree_0 = r0;
  Term _rest_0 = r1;
  WL_SPIN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      o[0] = _rest_0;
    } else {
      Term _fb_0[6];
      u64 _sp_0 = ctr_take(e, _tree_0, 6, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _f_2 = _fb_0[2];
      Term _f_3 = _fb_0[3];
      u32 _f_4 = _fb_0[4];
      u32 _f_5 = _fb_0[5];
      _f_0 = term_keep(e, _f_0, 1);
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 3);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, _f_1);
      e.mem[_nd_0 + 2] = rfc_seal(e, _f_2);
      e.mem[_nd_0 + 3] = rfc_seal(e, _f_3);
      e.mem[_nd_0 + 4] = rfc_seal(e, _f_4);
      e.mem[_nd_0 + 5] = rfc_seal(e, _f_5);
      u64 _nd_1 = heap_alloc(e, 1);
      e.mem[_nd_1 + 0] = rfc_seal(e, term_ctr(CID_ir_FunctionNode, _nd_0));
      e.mem[_nd_1 + 1] = rfc_seal(e, _rest_0);
      r0 = _f_0;
      r1 = term_ctr(CID_Con, _nd_1);
      _tree_0 = r0;
      _rest_0 = r1;
      WL_AGAIN(spin_ir_functionStack);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionHeight(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _tree_0 = r0;
  WL_SPIN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      o[0] = 0ull;
    } else {
      u64 _sp_0 = term_peek(e.mem, _tree_0);
      Term _f_0 = e.mem[_sp_0 + 0];
      Term _f_1 = e.mem[_sp_0 + 1];
      Term _f_2 = e.mem[_sp_0 + 2];
      Term _f_3 = e.mem[_sp_0 + 3];
      u32 _f_4 = e.mem[_sp_0 + 4];
      u32 _f_5 = e.mem[_sp_0 + 5];
      o[0] = _f_5;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionAppendBalance(Env e, THR Term* o, u32 r0, Term r1, Term r2, Term r3, Term r4) {
  u32 wpoll = 0;
  u32 _rotate_0 = r0;
  Term _left_0 = r1;
  Term _key_0 = r2;
  Term _value_0 = r3;
  Term _right_0 = r4;
  WL_SPIN
    if (_rotate_0 == 1) {
      if (term_aux(_right_0) == CID_ir_FunctionNode) {
        Term _fb_0[6];
        u64 _sp_0 = ctr_take(e, _right_0, 6, _fb_0);
        Term _f_0 = _fb_0[0];
        Term _f_1 = _fb_0[1];
        Term _f_2 = _fb_0[2];
        Term _f_3 = _fb_0[3];
        u32 _f_4 = _fb_0[4];
        u32 _f_5 = _fb_0[5];
        Term _o_0[1];
        if (spin_ir_functionNode(e, _o_0, _left_0, _key_0, _value_0, _f_0) == 0) {
          return 0;
        }
        Term _o_1[1];
        if (spin_ir_functionNode(e, _o_1, _o_0[0], _f_1, _f_2, _f_3) == 0) {
          return 0;
        }
        o[0] = _o_1[0];
        spare_free(e, cls_fit(6), _sp_0);
      } else {
        Term _o_2[1];
        if (spin_ir_functionNode(e, _o_2, _left_0, _key_0, _value_0, _right_0) == 0) {
          return 0;
        }
        o[0] = _o_2[0];
      }
    } else {
      Term _o_3[1];
      if (spin_ir_functionNode(e, _o_3, _left_0, _key_0, _value_0, _right_0) == 0) {
        return 0;
      }
      o[0] = _o_3[0];
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_sle(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_slt(e, _o_0, _a_0, _b_0) == 0) {
      return 0;
    }
    Term _a_1 = U32_BIN(_a_0, ==, _b_0);
    o[0] = (((u32)_o_0[0]) | (_a_1));
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionSize(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _tree_0 = r0;
  WL_SPIN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      o[0] = 0ull;
    } else {
      u64 _sp_0 = term_peek(e.mem, _tree_0);
      Term _f_0 = e.mem[_sp_0 + 0];
      Term _f_1 = e.mem[_sp_0 + 1];
      Term _f_2 = e.mem[_sp_0 + 2];
      Term _f_3 = e.mem[_sp_0 + 3];
      u32 _f_4 = e.mem[_sp_0 + 4];
      u32 _f_5 = e.mem[_sp_0 + 5];
      o[0] = _f_4;
    }
  break;
  }
  return 1;
}

INLINE Term spin_Bool_and(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    if (_a_0 == 0) {
      o[0] = 0;
    } else {
      o[0] = _b_0;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_buildFunctionRight(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3, Term r4, Term r5) {
  u32 wpoll = 0;
  Term _built_0 = r0;
  Term _built_1 = r1;
  Term _built_2 = r2;
  Term _left_0 = r3;
  Term _key_0 = r4;
  Term _value_0 = r5;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_functionNode(e, _o_0, _left_0, _key_0, _value_0, _built_0) == 0) {
      return 0;
    }
    o[0] = _o_0[0];
    o[1] = _built_1;
    o[2] = _built_2;
  break;
  }
  return 1;
}

INLINE Term spin_List_reverse(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _xs_0 = r0;
  WL_SPIN
    Term _o_0[1];
    if (spin_List_reverse_go(e, _o_0, _xs_0, term_pak(CID_Nil, 0)) == 0) {
      return 0;
    }
    o[0] = _o_0[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionAppendBalanced(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3) {
  u32 wpoll = 0;
  Term _left_0 = r0;
  Term _key_0 = r1;
  Term _value_0 = r2;
  Term _right_0 = r3;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_functionHeight(e, _o_0, _right_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_ir_functionHeight(e, _o_1, _left_0) == 0) {
      return 0;
    }
    Term _a_0 = U32_BIN((u32)_o_1[0], +, 1);
    Term _o_2[1];
    if (spin_ir_functionAppendBalance(e, _o_2, U32_BIN((u32)_o_0[0], >, _a_0), _left_0, _key_0, _value_0, _right_0) == 0) {
      return 0;
    }
    o[0] = _o_2[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_slt(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_negative(e, _o_0, _a_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_ir_negative(e, _o_1, _b_0) == 0) {
      return 0;
    }
    Term _o_2[1];
    if (spin_ir_slt_go(e, _o_2, (u32)_o_0[0], (u32)_o_1[0], _a_0, _b_0) == 0) {
      return 0;
    }
    o[0] = (u32)_o_2[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_emptyFunction(Env e, THR Term* o) {
  u32 wpoll = 0;
  WL_SPIN
    o[0] = term_ctr(CID_ir_VFun, STAT_OFF + 160);
  break;
  }
  return 1;
}

INLINE Term spin_ir_sge(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_sle(e, _o_0, _b_0, _a_0) == 0) {
      return 0;
    }
    o[0] = (u32)_o_0[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_domainTreeRight(Env e, THR Term* o, Term r0, Term r1, u32 r2, Term r3, Term r4) {
  u32 wpoll = 0;
  Term _left_0 = r0;
  Term _value_0 = r1;
  u32 _size_0 = r2;
  Term _result_0 = r3;
  Term _result_1 = r4;
  WL_SPIN
    u64 _nd_0 = heap_alloc(e, 2);
    e.mem[_nd_0 + 0] = rfc_seal(e, _left_0);
    e.mem[_nd_0 + 1] = rfc_seal(e, _value_0);
    e.mem[_nd_0 + 2] = rfc_seal(e, _result_0);
    e.mem[_nd_0 + 3] = rfc_seal(e, _size_0);
    o[0] = term_ctr(CID_ir_DomainNode, _nd_0);
    o[1] = _result_1;
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionMax(Env e, THR Term* o, u32 r0, u32 r1, u32 r2) {
  u32 wpoll = 0;
  u32 _greater_0 = r0;
  u32 _a_0 = r1;
  u32 _b_0 = r2;
  WL_SPIN
    if (_greater_0 == 1) {
      o[0] = _a_0;
    } else {
      o[0] = _b_0;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_domainSize(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _domain_0 = r0;
  WL_SPIN
    if (term_aux(_domain_0) == CID_ir_IntDomain) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _domain_0, 2, _fb_0);
      u32 _f_0 = _fb_0[0];
      u32 _f_1 = _fb_0[1];
      o[0] = _f_1;
      spare_free(e, cls_fit(2), _sp_0);
    } else {
      Term _fb_1[1];
      u64 _sp_1 = ctr_take(e, _domain_0, 1, _fb_1);
      Term _f_2 = _fb_1[0];
      Term _o_0[1];
      if (spin_ir_domainTreeSize(e, _o_0, _f_2) == 0) {
        return 0;
      }
      term_sink(e, _f_2);
      o[0] = (u32)_o_0[0];
      spare_free(e, cls_fit(1), _sp_1);
    }
  break;
  }
  return 1;
}

INLINE Term spin_List_reverse_go(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _xs_0 = r0;
  Term _acc_0 = r1;
  WL_SPIN
    if (term_aux(_xs_0) == CID_Nil) {
      o[0] = _acc_0;
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, _acc_0);
      r0 = _f_1;
      r1 = term_ctr(CID_Con, _nd_0);
      _xs_0 = r0;
      _acc_0 = r1;
      WL_AGAIN(spin_List_reverse_go);
    }
  break;
  }
  return 1;
}

INLINE Term spin_List_merge_step(Env e, THR Term* o, Term r0, Term r1, Term r2, Term r3, Term r4, u32 r5) {
  u32 wpoll = 0;
  Term _acc_0 = r0;
  Term _x_0 = r1;
  Term _xt_0 = r2;
  Term _y_0 = r3;
  Term _yt_0 = r4;
  u32 _le_0 = r5;
  WL_SPIN
    if (_le_0 == 0) {
      u64 _nd_0 = heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _y_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, _acc_0);
      u64 _nd_1 = heap_alloc(e, 1);
      e.mem[_nd_1 + 0] = rfc_seal(e, _x_0);
      e.mem[_nd_1 + 1] = rfc_seal(e, _xt_0);
      o[0] = term_ctr(CID_Con, _nd_0);
      o[1] = term_ctr(CID_Con, _nd_1);
      o[2] = _yt_0;
    } else {
      u64 _nd_2 = heap_alloc(e, 1);
      e.mem[_nd_2 + 0] = rfc_seal(e, _x_0);
      e.mem[_nd_2 + 1] = rfc_seal(e, _acc_0);
      u64 _nd_3 = heap_alloc(e, 1);
      e.mem[_nd_3 + 0] = rfc_seal(e, _y_0);
      e.mem[_nd_3 + 1] = rfc_seal(e, _yt_0);
      o[0] = term_ctr(CID_Con, _nd_2);
      o[1] = _xt_0;
      o[2] = term_ctr(CID_Con, _nd_3);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_sgt(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_slt(e, _o_0, _b_0, _a_0) == 0) {
      return 0;
    }
    o[0] = (u32)_o_0[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_domainBuilt(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _result_0 = r0;
  Term _result_1 = r1;
  WL_SPIN
    term_sink(e, _result_1);
    u64 _nd_0 = heap_alloc(e, 0);
    e.mem[_nd_0 + 0] = rfc_seal(e, _result_0);
    o[0] = term_ctr(CID_ir_TreeDomain, _nd_0);
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionBuiltValue(Env e, THR Term* o, Term r0, Term r1, Term r2, u32 r3) {
  u32 wpoll = 0;
  Term _built_0 = r0;
  Term _built_1 = r1;
  Term _built_2 = r2;
  u32 _sequence_0 = r3;
  WL_SPIN
    if (term_aux(_built_1) == CID_Nil) {
      if (term_aux(_built_2) == CID_Nil) {
        u64 _nd_0 = heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _built_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, _sequence_0);
        o[0] = term_ctr(CID_ir_VFun, _nd_0);
      } else {
        term_sink(e, _built_0);
        term_sink(e, _built_2);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 228);
      }
    } else {
      term_sink(e, _built_0);
      term_sink(e, _built_1);
      term_sink(e, _built_2);
      o[0] = term_ctr(CID_ir_VE, STAT_OFF + 228);
    }
  break;
  }
  return 1;
}

INLINE Term spin_Cmp_is_le(Env e, THR Term* o, u32 r0) {
  u32 wpoll = 0;
  u32 _c_0 = r0;
  WL_SPIN
    if (_c_0 == 2) {
      o[0] = 0;
    } else {
      o[0] = 1;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionFound(Env e, THR Term* o, u32 r0, Term r1) {
  u32 wpoll = 0;
  u32 _value_0 = r0;
  Term _value_1 = r1;
  WL_SPIN
    if (_value_0 == 1) {
      o[0] = _value_1;
    } else {
      o[0] = term_ctr(CID_ir_VE, STAT_OFF + 1258);
    }
  break;
  }
  return 1;
}

INLINE Term spin_String_order_go(Env e, THR Term* o, Term r0, Term r1, u32 r2) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  u32 _c_0 = r2;
  WL_SPIN
    if (term_aux(_a_0) == CID_SNil) {
      if (term_aux(_b_0) == CID_SCon) {
        Term _fb_0[2];
        u64 _sp_0 = ctr_take(e, _b_0, 2, _fb_0);
        u32 _f_0 = _fb_0[0];
        Term _f_1 = _fb_0[1];
        term_sink(e, _f_1);
        if (_c_0 == 1) {
          o[0] = 0;
          spare_free(e, cls_fit(2), _sp_0);
        } else {
          o[0] = _c_0;
          spare_free(e, cls_fit(2), _sp_0);
        }
      } else {
        term_sink(e, _b_0);
        o[0] = _c_0;
      }
    } else {
      Term _fb_1[2];
      u64 _sp_1 = ctr_take(e, _a_0, 2, _fb_1);
      u32 _f_2 = _fb_1[0];
      Term _f_3 = _fb_1[1];
      if (term_aux(_b_0) == CID_SNil) {
        term_sink(e, _f_3);
        if (_c_0 == 1) {
          o[0] = 2;
          spare_free(e, cls_fit(2), _sp_1);
        } else {
          o[0] = _c_0;
          spare_free(e, cls_fit(2), _sp_1);
        }
      } else {
        Term _fb_2[2];
        u64 _sp_2 = ctr_take(e, _b_0, 2, _fb_2);
        u32 _f_4 = _fb_2[0];
        Term _f_5 = _fb_2[1];
        if (_c_0 == 1) {
          spare_free(e, cls_fit(2), _sp_2);
          spare_free(e, cls_fit(2), _sp_1);
          r0 = _f_3;
          r1 = _f_5;
          r2 = (U32_BIN(_f_2, >, _f_4) + U32_BIN(_f_2, >=, _f_4));
          _a_0 = r0;
          _b_0 = r1;
          _c_0 = r2;
          WL_AGAIN(spin_String_order_go);
        } else {
          term_sink(e, _f_3);
          term_sink(e, _f_5);
          o[0] = _c_0;
          spare_free(e, cls_fit(2), _sp_2);
          spare_free(e, cls_fit(2), _sp_1);
        }
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_negative(Env e, THR Term* o, u32 r0) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  WL_SPIN
    o[0] = U32_BIN(_a_0, >=, 2147483648ull);
  break;
  }
  return 1;
}

INLINE Term spin_ir_signedOrderSigns(Env e, THR Term* o, u32 r0, u32 r1, u32 r2, u32 r3) {
  u32 wpoll = 0;
  u32 _a_negative_0 = r0;
  u32 _b_negative_0 = r1;
  u32 _a_0 = r2;
  u32 _b_0 = r3;
  WL_SPIN
    if (_a_negative_0 == 1) {
      if (_b_negative_0 == 0) {
        o[0] = 0;
      } else {
        o[0] = (U32_BIN(_a_0, >, _b_0) + U32_BIN(_a_0, >=, _b_0));
      }
    } else {
      if (_b_negative_0 == 1) {
        o[0] = 2;
      } else {
        o[0] = (U32_BIN(_a_0, >, _b_0) + U32_BIN(_a_0, >=, _b_0));
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_domainTreeSize(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _tree_0 = r0;
  WL_SPIN
    if (term_aux(_tree_0) == CID_ir_DomainEmpty) {
      o[0] = 0ull;
    } else {
      u64 _sp_0 = term_peek(e.mem, _tree_0);
      Term _f_0 = e.mem[_sp_0 + 0];
      Term _f_1 = e.mem[_sp_0 + 1];
      Term _f_2 = e.mem[_sp_0 + 2];
      u32 _f_3 = e.mem[_sp_0 + 3];
      o[0] = _f_3;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_domainRangeRank(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _valid_0 = r0;
  u32 _index_0 = r1;
  WL_SPIN
    if (_valid_0 == 1) {
      o[0] = 1;
      o[1] = _index_0;
    } else {
      o[0] = 0;
      o[1] = 0;
    }
  break;
  }
  return 1;
}

INLINE Term spin_List_sort_pair(Env e, THR Term* o, Term r0, Term r1, u32 r2) {
  u32 wpoll = 0;
  Term _x_0 = r0;
  Term _y_0 = r1;
  u32 _le_0 = r2;
  WL_SPIN
    if (_le_0 == 1) {
      u64 _nd_0 = heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _y_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, term_pak(CID_Nil, 0));
      u64 _nd_1 = heap_alloc(e, 1);
      e.mem[_nd_1 + 0] = rfc_seal(e, _x_0);
      e.mem[_nd_1 + 1] = rfc_seal(e, term_ctr(CID_Con, _nd_0));
      o[0] = term_ctr(CID_Con, _nd_1);
    } else {
      u64 _nd_2 = heap_alloc(e, 1);
      e.mem[_nd_2 + 0] = rfc_seal(e, _x_0);
      e.mem[_nd_2 + 1] = rfc_seal(e, term_pak(CID_Nil, 0));
      u64 _nd_3 = heap_alloc(e, 1);
      e.mem[_nd_3 + 0] = rfc_seal(e, _y_0);
      e.mem[_nd_3 + 1] = rfc_seal(e, term_ctr(CID_Con, _nd_2));
      o[0] = term_ctr(CID_Con, _nd_3);
    }
  break;
  }
  return 1;
}

INLINE Term spin_String_order(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_String_order_go(e, _o_0, _a_0, _b_0, 1) == 0) {
      return 0;
    }
    o[0] = (u32)_o_0[0];
  break;
  }
  return 1;
}

INLINE Term spin_Cmp_is_eq(Env e, THR Term* o, u32 r0) {
  u32 wpoll = 0;
  u32 _c_0 = r0;
  WL_SPIN
    if (_c_0 == 1) {
      o[0] = 1;
    } else {
      o[0] = 0;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_sequenceLengthArg(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _args_0 = r0;
  WL_SPIN
    if (term_aux(_args_0) == CID_Con) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _args_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VFun) {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _f_0, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        u32 _f_3 = _fb_1[1];
        if (_f_3 == 1) {
          if (term_aux(_f_1) == CID_Nil) {
            Term _o_0[1];
            if (spin_ir_functionSize(e, _o_0, _f_2) == 0) {
              return 0;
            }
            term_sink(e, _f_2);
            o[0] = term_pak(CID_ir_VI, (u32)_o_0[0]);
            spare_free(e, cls_fit(2), _sp_1);
            spare_free(e, cls_fit(2), _sp_0);
          } else {
            term_sink(e, _f_2);
            term_sink(e, _f_1);
            o[0] = term_ctr(CID_ir_VE, STAT_OFF + 791);
            spare_free(e, cls_fit(2), _sp_1);
            spare_free(e, cls_fit(2), _sp_0);
          }
        } else {
          term_sink(e, _f_2);
          term_sink(e, _f_1);
          o[0] = term_ctr(CID_ir_VE, STAT_OFF + 791);
          spare_free(e, cls_fit(2), _sp_1);
          spare_free(e, cls_fit(2), _sp_0);
        }
      } else {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 791);
        spare_free(e, cls_fit(2), _sp_0);
      }
    } else {
      term_sink(e, _args_0);
      o[0] = term_ctr(CID_ir_VE, STAT_OFF + 791);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_signedOrder(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_negative(e, _o_0, _a_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_ir_negative(e, _o_1, _b_0) == 0) {
      return 0;
    }
    Term _o_2[1];
    if (spin_ir_signedOrderSigns(e, _o_2, (u32)_o_0[0], (u32)_o_1[0], _a_0, _b_0) == 0) {
      return 0;
    }
    o[0] = (u32)_o_2[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_valueTag(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _value_0 = r0;
  WL_SPIN
    if (term_aux(_value_0) == CID_ir_VI) {
      u32 __0 = term_loc(_value_0);
      o[0] = 0ull;
    } else if (term_aux(_value_0) == CID_ir_VB) {
      u32 __1 = term_loc(_value_0);
      o[0] = 1ull;
    } else if (term_aux(_value_0) == CID_ir_VS) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _value_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      term_sink(e, _f_0);
      o[0] = 2ull;
      spare_free(e, cls_fit(1), _sp_0);
    } else if (term_aux(_value_0) == CID_ir_VSet) {
      Term _fb_1[2];
      u64 _sp_1 = ctr_take(e, _value_0, 2, _fb_1);
      Term _f_1 = _fb_1[0];
      Term _f_2 = _fb_1[1];
      term_sink(e, _f_1);
      term_sink(e, _f_2);
      o[0] = 3ull;
      spare_free(e, cls_fit(2), _sp_1);
    } else if (term_aux(_value_0) == CID_ir_VFun) {
      Term _fb_2[2];
      u64 _sp_2 = ctr_take(e, _value_0, 2, _fb_2);
      Term _f_3 = _fb_2[0];
      u32 _f_4 = _fb_2[1];
      term_sink(e, _f_3);
      o[0] = 4ull;
      spare_free(e, cls_fit(2), _sp_2);
    } else {
      Term _fb_3[1];
      u64 _sp_3 = ctr_take(e, _value_0, 1, _fb_3);
      Term _f_5 = _fb_3[0];
      term_sink(e, _f_5);
      o[0] = 5ull;
      spare_free(e, cls_fit(1), _sp_3);
    }
  break;
  }
  return 1;
}

INLINE Term spin_Bool_to_u32(Env e, THR Term* o, u32 r0) {
  u32 wpoll = 0;
  u32 _b_0 = r0;
  WL_SPIN
    if (_b_0 == 0) {
      o[0] = 0ull;
    } else {
      o[0] = 1ull;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_emptySet(Env e, THR Term* o) {
  u32 wpoll = 0;
  WL_SPIN
    o[0] = term_ctr(CID_ir_VSet, STAT_OFF + 795);
  break;
  }
  return 1;
}

INLINE Term spin_ir_domainRangeIndex(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _index_0 = r0;
  u32 _size_0 = r1;
  WL_SPIN
    Term _o_0[2];
    if (spin_ir_domainRangeRank(e, _o_0, U32_BIN(_index_0, <, _size_0), _index_0) == 0) {
      return 0;
    }
    o[0] = (u32)_o_0[0];
    o[1] = (u32)_o_0[1];
  break;
  }
  return 1;
}

INLINE Term spin_Bool_not(Env e, THR Term* o, u32 r0) {
  u32 wpoll = 0;
  u32 _b_0 = r0;
  WL_SPIN
    if (_b_0 == 0) {
      o[0] = 1;
    } else {
      o[0] = 0;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_slt_go(Env e, THR Term* o, u32 r0, u32 r1, u32 r2, u32 r3) {
  u32 wpoll = 0;
  u32 _na_0 = r0;
  u32 _nb_0 = r1;
  u32 _a_0 = r2;
  u32 _b_0 = r3;
  WL_SPIN
    if (_na_0 == 1) {
      if (_nb_0 == 0) {
        o[0] = 1;
      } else {
        o[0] = U32_BIN(_a_0, <, _b_0);
      }
    } else {
      if (_nb_0 == 1) {
        o[0] = 0;
      } else {
        o[0] = U32_BIN(_a_0, <, _b_0);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_boolEq(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    if (_a_0 == 1) {
      if (_b_0 == 1) {
        o[0] = 1;
      } else {
        o[0] = 0;
      }
    } else {
      if (_b_0 == 0) {
        o[0] = 1;
      } else {
        o[0] = 0;
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_String_eq(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_String_order(e, _o_0, _a_0, _b_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_Cmp_is_eq(e, _o_1, (u32)_o_0[0]) == 0) {
      return 0;
    }
    o[0] = (u32)_o_1[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_functionEqual(Env e, THR Term* o, u32 r0) {
  u32 wpoll = 0;
  u32 _order_0 = r0;
  WL_SPIN
    if (_order_0 == 1) {
      o[0] = 1;
    } else {
      o[0] = 0;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_pickVal(Env e, THR Term* o, u32 r0, Term r1, Term r2) {
  u32 wpoll = 0;
  u32 _c_0 = r0;
  Term _yes_0 = r1;
  Term _no_0 = r2;
  WL_SPIN
    if (_c_0 == 1) {
      term_sink(e, _no_0);
      o[0] = _yes_0;
    } else {
      term_sink(e, _yes_0);
      o[0] = _no_0;
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_bindLocal(Env e, THR Term* o, Term r0, Term r1, u32 r2, Term r3) {
  u32 wpoll = 0;
  Term _env_0 = r0;
  Term _env_1 = r1;
  u32 _slot_0 = r2;
  Term _value_0 = r3;
  WL_SPIN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _slot_0);
    e.mem[_nd_0 + 1] = rfc_seal(e, _value_0);
    u64 _nd_1 = heap_alloc(e, 1);
    e.mem[_nd_1 + 0] = rfc_seal(e, term_ctr(CID_ir_Local, _nd_0));
    e.mem[_nd_1 + 1] = rfc_seal(e, _env_1);
    o[0] = _env_0;
    o[1] = term_ctr(CID_Con, _nd_1);
  break;
  }
  return 1;
}

INLINE Term spin_ir_productCountEnd(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _fits_0 = r0;
  u32 _count_0 = r1;
  WL_SPIN
    if (_fits_0 == 1) {
      o[0] = term_pak(CID_ir_VI, _count_0);
    } else {
      o[0] = term_ctr(CID_ir_VE, STAT_OFF + 979);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_memberResult(Env e, THR Term* o, u32 r0, u32 r1, u32 r2) {
  u32 wpoll = 0;
  u32 _found_0 = r0;
  u32 _found_1 = r1;
  u32 _negate_0 = r2;
  WL_SPIN
    if (_found_0 == 0) {
      o[0] = term_pak(CID_ir_VB, _negate_0);
    } else {
      Term _o_0[1];
      if (spin_Bool_not(e, _o_0, _negate_0) == 0) {
        return 0;
      }
      o[0] = term_pak(CID_ir_VB, (u32)_o_0[0]);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_booleanValue(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _value_0 = r0;
  WL_SPIN
    if (term_aux(_value_0) == CID_ir_VB) {
      u32 _b_0 = term_loc(_value_0);
      o[0] = term_pak(CID_ir_VB, _b_0);
    } else if (term_aux(_value_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _value_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      o[0] = term_ctr(CID_ir_VE, _nd_0);
    } else {
      term_sink(e, _value_0);
      o[0] = term_ctr(CID_ir_VE, STAT_OFF + 1090);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_vle(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    if (term_aux(_a_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _b_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_0);
        spare_free(e, cls_fit(1), _sp_1);
      } else {
        term_sink(e, _b_0);
        u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_1);
      }
    } else if (term_aux(_a_0) == CID_ir_VI) {
      u32 _x_0 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_2[1];
        u64 _sp_2 = ctr_take(e, _b_0, 1, _fb_2);
        Term _f_2 = _fb_2[0];
        u64 _nd_2 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 0);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_2);
        o[0] = term_ctr(CID_ir_VE, _nd_2);
      } else if (term_aux(_b_0) == CID_ir_VI) {
        u32 _y_0 = term_loc(_b_0);
        Term _o_0[1];
        if (spin_ir_sle(e, _o_0, _x_0, _y_0) == 0) {
          return 0;
        }
        o[0] = term_pak(CID_ir_VB, (u32)_o_0[0]);
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2023);
      }
    } else {
      term_sink(e, _a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_3[1];
        u64 _sp_3 = ctr_take(e, _b_0, 1, _fb_3);
        Term _f_3 = _fb_3[0];
        u64 _nd_3 = _sp_3 >= HEAP_OFF ? _sp_3 : heap_alloc(e, 0);
        e.mem[_nd_3 + 0] = rfc_seal(e, _f_3);
        o[0] = term_ctr(CID_ir_VE, _nd_3);
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2023);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_vlt(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    if (term_aux(_a_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _b_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_0);
        spare_free(e, cls_fit(1), _sp_1);
      } else {
        term_sink(e, _b_0);
        u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_1);
      }
    } else if (term_aux(_a_0) == CID_ir_VI) {
      u32 _x_0 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_2[1];
        u64 _sp_2 = ctr_take(e, _b_0, 1, _fb_2);
        Term _f_2 = _fb_2[0];
        u64 _nd_2 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 0);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_2);
        o[0] = term_ctr(CID_ir_VE, _nd_2);
      } else if (term_aux(_b_0) == CID_ir_VI) {
        u32 _y_0 = term_loc(_b_0);
        Term _o_0[1];
        if (spin_ir_slt(e, _o_0, _x_0, _y_0) == 0) {
          return 0;
        }
        o[0] = term_pak(CID_ir_VB, (u32)_o_0[0]);
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2070);
      }
    } else {
      term_sink(e, _a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_3[1];
        u64 _sp_3 = ctr_take(e, _b_0, 1, _fb_3);
        Term _f_3 = _fb_3[0];
        u64 _nd_3 = _sp_3 >= HEAP_OFF ? _sp_3 : heap_alloc(e, 0);
        e.mem[_nd_3 + 0] = rfc_seal(e, _f_3);
        o[0] = term_ctr(CID_ir_VE, _nd_3);
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2070);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_vnot(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  WL_SPIN
    if (term_aux(_a_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      o[0] = term_ctr(CID_ir_VE, _nd_0);
    } else if (term_aux(_a_0) == CID_ir_VB) {
      u32 _x_0 = term_loc(_a_0);
      Term _o_0[1];
      if (spin_Bool_not(e, _o_0, _x_0) == 0) {
        return 0;
      }
      o[0] = term_pak(CID_ir_VB, (u32)_o_0[0]);
    } else {
      term_sink(e, _a_0);
      o[0] = term_ctr(CID_ir_VE, STAT_OFF + 1876);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_sneg(Env e, THR Term* o, u32 r0) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_pickVal(e, _o_0, U32_BIN(_a_0, ==, 2147483648ull), term_ctr(CID_ir_VE, STAT_OFF + 1153), term_pak(CID_ir_VI, U32_BIN(0ull, -, _a_0))) == 0) {
      return 0;
    }
    o[0] = _o_0[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_ssub(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    u32 _difference_0 = U32_BIN(_a_0, -, _b_0);
    Term _o_0[1];
    if (spin_ir_negative(e, _o_0, _a_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_ir_negative(e, _o_1, _b_0) == 0) {
      return 0;
    }
    u32 _different_signs_0 = (((u32)_o_0[0]) ^ ((u32)_o_1[0]));
    Term _o_2[1];
    if (spin_ir_negative(e, _o_2, _difference_0) == 0) {
      return 0;
    }
    Term _o_3[1];
    if (spin_ir_negative(e, _o_3, _a_0) == 0) {
      return 0;
    }
    u32 _changed_sign_0 = (((u32)_o_2[0]) ^ ((u32)_o_3[0]));
    Term _o_4[1];
    if (spin_Bool_and(e, _o_4, _different_signs_0, _changed_sign_0) == 0) {
      return 0;
    }
    Term _o_5[1];
    if (spin_ir_pickVal(e, _o_5, (u32)_o_4[0], term_ctr(CID_ir_VE, STAT_OFF + 1196), term_pak(CID_ir_VI, _difference_0)) == 0) {
      return 0;
    }
    o[0] = _o_5[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_sadd(Env e, THR Term* o, u32 r0, u32 r1) {
  u32 wpoll = 0;
  u32 _a_0 = r0;
  u32 _b_0 = r1;
  WL_SPIN
    u32 _s_0 = U32_BIN(_a_0, +, _b_0);
    Term _o_0[1];
    if (spin_ir_negative(e, _o_0, _a_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_ir_negative(e, _o_1, _b_0) == 0) {
      return 0;
    }
    Term _o_2[1];
    if (spin_Bool_not(e, _o_2, (((u32)_o_0[0]) ^ ((u32)_o_1[0]))) == 0) {
      return 0;
    }
    Term _o_3[1];
    if (spin_ir_negative(e, _o_3, _s_0) == 0) {
      return 0;
    }
    Term _o_4[1];
    if (spin_ir_negative(e, _o_4, _a_0) == 0) {
      return 0;
    }
    u32 _flipped_0 = (((u32)_o_3[0]) ^ ((u32)_o_4[0]));
    Term _o_5[1];
    if (spin_Bool_and(e, _o_5, (u32)_o_2[0], _flipped_0) == 0) {
      return 0;
    }
    Term _o_6[1];
    if (spin_ir_pickVal(e, _o_6, (u32)_o_5[0], term_ctr(CID_ir_VE, STAT_OFF + 1233), term_pak(CID_ir_VI, _s_0)) == 0) {
      return 0;
    }
    o[0] = _o_6[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_setInputErrorValue(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _values_0 = r0;
  WL_SPIN
    if (term_aux(_values_0) == CID_Nil) {
      o[0] = 0;
      o[1] = 0;
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _f_0, 1, _fb_1);
        Term _f_2 = _fb_1[0];
        term_sink(e, _f_1);
        o[0] = 1;
        o[1] = _f_2;
        spare_free(e, cls_fit(1), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
      } else {
        term_sink(e, _f_0);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_1;
        _values_0 = r0;
        WL_AGAIN(spin_ir_setInputErrorValue);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_nthVal(Env e, THR Term* o, Term r0, Term r1, Term r2) {
  u32 wpoll = 0;
  Term _xs_0 = r0;
  Term _n_0 = r1;
  Term _d_0 = r2;
  WL_SPIN
    if (term_aux(_xs_0) == CID_Nil) {
      o[0] = _d_0;
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (_n_0 == 0) {
        term_sink(e, _f_1);
        term_sink(e, _d_0);
        o[0] = _f_0;
        spare_free(e, cls_fit(2), _sp_0);
      } else {
        term_sink(e, _f_0);
        Term _p_0 = (_n_0 - 1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_1;
        r1 = _p_0;
        r2 = _d_0;
        _xs_0 = r0;
        _n_0 = r1;
        _d_0 = r2;
        WL_AGAIN(spin_ir_nthVal);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_vadd(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    if (term_aux(_a_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _b_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_0);
        spare_free(e, cls_fit(1), _sp_1);
      } else {
        term_sink(e, _b_0);
        u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_1);
      }
    } else if (term_aux(_a_0) == CID_ir_VI) {
      u32 _x_0 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_2[1];
        u64 _sp_2 = ctr_take(e, _b_0, 1, _fb_2);
        Term _f_2 = _fb_2[0];
        u64 _nd_2 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 0);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_2);
        o[0] = term_ctr(CID_ir_VE, _nd_2);
      } else if (term_aux(_b_0) == CID_ir_VI) {
        u32 _y_0 = term_loc(_b_0);
        Term _o_0[1];
        if (spin_ir_sadd(e, _o_0, _x_0, _y_0) == 0) {
          return 0;
        }
        o[0] = _o_0[0];
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2223);
      }
    } else {
      term_sink(e, _a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_3[1];
        u64 _sp_3 = ctr_take(e, _b_0, 1, _fb_3);
        Term _f_3 = _fb_3[0];
        u64 _nd_3 = _sp_3 >= HEAP_OFF ? _sp_3 : heap_alloc(e, 0);
        e.mem[_nd_3 + 0] = rfc_seal(e, _f_3);
        o[0] = term_ctr(CID_ir_VE, _nd_3);
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2223);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_vsub(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    if (term_aux(_a_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _b_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_0);
        spare_free(e, cls_fit(1), _sp_1);
      } else {
        term_sink(e, _b_0);
        u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
        o[0] = term_ctr(CID_ir_VE, _nd_1);
      }
    } else if (term_aux(_a_0) == CID_ir_VI) {
      u32 _x_0 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_2[1];
        u64 _sp_2 = ctr_take(e, _b_0, 1, _fb_2);
        Term _f_2 = _fb_2[0];
        u64 _nd_2 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 0);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_2);
        o[0] = term_ctr(CID_ir_VE, _nd_2);
      } else if (term_aux(_b_0) == CID_ir_VI) {
        u32 _y_0 = term_loc(_b_0);
        Term _o_0[1];
        if (spin_ir_ssub(e, _o_0, _x_0, _y_0) == 0) {
          return 0;
        }
        o[0] = _o_0[0];
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2176);
      }
    } else {
      term_sink(e, _a_0);
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_3[1];
        u64 _sp_3 = ctr_take(e, _b_0, 1, _fb_3);
        Term _f_3 = _fb_3[0];
        u64 _nd_3 = _sp_3 >= HEAP_OFF ? _sp_3 : heap_alloc(e, 0);
        e.mem[_nd_3 + 0] = rfc_seal(e, _f_3);
        o[0] = term_ctr(CID_ir_VE, _nd_3);
      } else {
        term_sink(e, _b_0);
        o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2176);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_vneg(Env e, THR Term* o, Term r0) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  WL_SPIN
    if (term_aux(_a_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      o[0] = term_ctr(CID_ir_VE, _nd_0);
    } else if (term_aux(_a_0) == CID_ir_VI) {
      u32 _x_0 = term_loc(_a_0);
      Term _o_0[1];
      if (spin_ir_sneg(e, _o_0, _x_0) == 0) {
        return 0;
      }
      o[0] = _o_0[0];
    } else {
      term_sink(e, _a_0);
      o[0] = term_ctr(CID_ir_VE, STAT_OFF + 2133);
    }
  break;
  }
  return 1;
}

INLINE Term spin_ir_vgt(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_vlt(e, _o_0, _b_0, _a_0) == 0) {
      return 0;
    }
    o[0] = _o_0[0];
  break;
  }
  return 1;
}

INLINE Term spin_ir_vge(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _a_0 = r0;
  Term _b_0 = r1;
  WL_SPIN
    Term _o_0[1];
    if (spin_ir_vle(e, _o_0, _b_0, _a_0) == 0) {
      return 0;
    }
    o[0] = _o_0[0];
  break;
  }
  return 1;
}

DFAR Term spin_ir_nthNode(Env e, THR Term* o, Term r0, Term r1) {
  u32 wpoll = 0;
  Term _xs_0 = r0;
  Term _n_0 = r1;
  WL_SPIN
    if (term_aux(_xs_0) == CID_Nil) {
      o[0] = 30;
      o[1] = term_ctr(CID_SCon, STAT_OFF + 2457);
      o[2] = 0;
      o[3] = 0;
      o[4] = 0;
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      u32 _o_0 = 0;
      Term _o_1 = 0;
      Term _o_2 = 0;
      u32 _o_3 = 0;
      u32 _o_4 = 0;
      if (term_aux(_f_0) == CID_ir_IInt) {
        _o_0 = 0;
        _o_1 = term_loc(_f_0);
      } else if (term_aux(_f_0) == CID_ir_IBool) {
        _o_0 = 1;
        _o_1 = term_loc(_f_0);
      } else if (term_aux(_f_0) == CID_ir_IStr) {
        _o_0 = 2;
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _f_0, 1, _fb_1);
        Term _f_2 = _fb_1[0];
        spare_free(e, cls_fit(1), _sp_1);
        _o_1 = _f_2;
      } else if (term_aux(_f_0) == CID_ir_ISlot) {
        _o_0 = 3;
        _o_1 = term_loc(_f_0);
      } else if (term_aux(_f_0) == CID_ir_IValue) {
        _o_0 = 4;
        Term _fb_2[1];
        u64 _sp_2 = ctr_take(e, _f_0, 1, _fb_2);
        Term _f_3 = _fb_2[0];
        spare_free(e, cls_fit(1), _sp_2);
        _o_1 = _f_3;
      } else if (term_aux(_f_0) == CID_ir_ILocal) {
        _o_0 = 5;
        _o_1 = term_loc(_f_0);
      } else if (term_aux(_f_0) == CID_ir_ISet) {
        _o_0 = 6;
        Term _fb_3[1];
        u64 _sp_3 = ctr_take(e, _f_0, 1, _fb_3);
        Term _f_4 = _fb_3[0];
        spare_free(e, cls_fit(1), _sp_3);
        _o_1 = _f_4;
      } else if (term_aux(_f_0) == CID_ir_ISetBinary) {
        _o_0 = 7;
        Term _fb_4[3];
        u64 _sp_4 = ctr_take(e, _f_0, 3, _fb_4);
        u32 _f_5 = _fb_4[0];
        u32 _f_6 = _fb_4[1];
        u32 _f_7 = _fb_4[2];
        spare_free(e, cls_fit(3), _sp_4);
        _o_1 = _f_5;
        _o_2 = _f_6;
        _o_3 = _f_7;
      } else if (term_aux(_f_0) == CID_ir_ISetUnary) {
        _o_0 = 8;
        Term _fb_5[2];
        u64 _sp_5 = ctr_take(e, _f_0, 2, _fb_5);
        u32 _f_8 = _fb_5[0];
        u32 _f_9 = _fb_5[1];
        spare_free(e, cls_fit(2), _sp_5);
        _o_1 = _f_8;
        _o_2 = _f_9;
      } else if (term_aux(_f_0) == CID_ir_IBounded) {
        _o_0 = 9;
        Term _fb_6[4];
        u64 _sp_6 = ctr_take(e, _f_0, 4, _fb_6);
        u32 _f_10 = _fb_6[0];
        u32 _f_11 = _fb_6[1];
        u32 _f_12 = _fb_6[2];
        u32 _f_13 = _fb_6[3];
        spare_free(e, cls_fit(4), _sp_6);
        _o_1 = _f_10;
        _o_2 = _f_11;
        _o_3 = _f_12;
        _o_4 = _f_13;
      } else if (term_aux(_f_0) == CID_ir_IFunction) {
        _o_0 = 10;
        Term _fb_7[2];
        u64 _sp_7 = ctr_take(e, _f_0, 2, _fb_7);
        Term _f_14 = _fb_7[0];
        Term _f_15 = _fb_7[1];
        spare_free(e, cls_fit(2), _sp_7);
        _o_1 = _f_14;
        _o_2 = _f_15;
      } else if (term_aux(_f_0) == CID_ir_IProduct) {
        _o_0 = 11;
        Term _fb_8[2];
        u64 _sp_8 = ctr_take(e, _f_0, 2, _fb_8);
        Term _f_16 = _fb_8[0];
        Term _f_17 = _fb_8[1];
        spare_free(e, cls_fit(2), _sp_8);
        _o_1 = _f_16;
        _o_2 = _f_17;
      } else if (term_aux(_f_0) == CID_ir_ILookup) {
        _o_0 = 12;
        Term _fb_9[2];
        u64 _sp_9 = ctr_take(e, _f_0, 2, _fb_9);
        u32 _f_18 = _fb_9[0];
        u32 _f_19 = _fb_9[1];
        spare_free(e, cls_fit(2), _sp_9);
        _o_1 = _f_18;
        _o_2 = _f_19;
      } else if (term_aux(_f_0) == CID_ir_IUpdate) {
        _o_0 = 13;
        Term _fb_10[4];
        u64 _sp_10 = ctr_take(e, _f_0, 4, _fb_10);
        u32 _f_20 = _fb_10[0];
        u32 _f_21 = _fb_10[1];
        u32 _f_22 = _fb_10[2];
        u32 _f_23 = _fb_10[3];
        spare_free(e, cls_fit(4), _sp_10);
        _o_1 = _f_20;
        _o_2 = _f_21;
        _o_3 = _f_22;
        _o_4 = _f_23;
      } else if (term_aux(_f_0) == CID_ir_IDomain) {
        _o_0 = 14;
        _o_1 = term_loc(_f_0);
      } else if (term_aux(_f_0) == CID_ir_ISequence) {
        _o_0 = 15;
        Term _fb_11[2];
        u64 _sp_11 = ctr_take(e, _f_0, 2, _fb_11);
        u32 _f_24 = _fb_11[0];
        Term _f_25 = _fb_11[1];
        spare_free(e, cls_fit(2), _sp_11);
        _o_1 = _f_24;
        _o_2 = _f_25;
      } else if (term_aux(_f_0) == CID_ir_IAdd) {
        _o_0 = 16;
        Term _fb_12[2];
        u64 _sp_12 = ctr_take(e, _f_0, 2, _fb_12);
        u32 _f_26 = _fb_12[0];
        u32 _f_27 = _fb_12[1];
        spare_free(e, cls_fit(2), _sp_12);
        _o_1 = _f_26;
        _o_2 = _f_27;
      } else if (term_aux(_f_0) == CID_ir_ISub) {
        _o_0 = 17;
        Term _fb_13[2];
        u64 _sp_13 = ctr_take(e, _f_0, 2, _fb_13);
        u32 _f_28 = _fb_13[0];
        u32 _f_29 = _fb_13[1];
        spare_free(e, cls_fit(2), _sp_13);
        _o_1 = _f_28;
        _o_2 = _f_29;
      } else if (term_aux(_f_0) == CID_ir_INeg) {
        _o_0 = 18;
        _o_1 = term_loc(_f_0);
      } else if (term_aux(_f_0) == CID_ir_IEq) {
        _o_0 = 19;
        Term _fb_14[2];
        u64 _sp_14 = ctr_take(e, _f_0, 2, _fb_14);
        u32 _f_30 = _fb_14[0];
        u32 _f_31 = _fb_14[1];
        spare_free(e, cls_fit(2), _sp_14);
        _o_1 = _f_30;
        _o_2 = _f_31;
      } else if (term_aux(_f_0) == CID_ir_INe) {
        _o_0 = 20;
        Term _fb_15[2];
        u64 _sp_15 = ctr_take(e, _f_0, 2, _fb_15);
        u32 _f_32 = _fb_15[0];
        u32 _f_33 = _fb_15[1];
        spare_free(e, cls_fit(2), _sp_15);
        _o_1 = _f_32;
        _o_2 = _f_33;
      } else if (term_aux(_f_0) == CID_ir_ILt) {
        _o_0 = 21;
        Term _fb_16[2];
        u64 _sp_16 = ctr_take(e, _f_0, 2, _fb_16);
        u32 _f_34 = _fb_16[0];
        u32 _f_35 = _fb_16[1];
        spare_free(e, cls_fit(2), _sp_16);
        _o_1 = _f_34;
        _o_2 = _f_35;
      } else if (term_aux(_f_0) == CID_ir_ILe) {
        _o_0 = 22;
        Term _fb_17[2];
        u64 _sp_17 = ctr_take(e, _f_0, 2, _fb_17);
        u32 _f_36 = _fb_17[0];
        u32 _f_37 = _fb_17[1];
        spare_free(e, cls_fit(2), _sp_17);
        _o_1 = _f_36;
        _o_2 = _f_37;
      } else if (term_aux(_f_0) == CID_ir_IGt) {
        _o_0 = 23;
        Term _fb_18[2];
        u64 _sp_18 = ctr_take(e, _f_0, 2, _fb_18);
        u32 _f_38 = _fb_18[0];
        u32 _f_39 = _fb_18[1];
        spare_free(e, cls_fit(2), _sp_18);
        _o_1 = _f_38;
        _o_2 = _f_39;
      } else if (term_aux(_f_0) == CID_ir_IGe) {
        _o_0 = 24;
        Term _fb_19[2];
        u64 _sp_19 = ctr_take(e, _f_0, 2, _fb_19);
        u32 _f_40 = _fb_19[0];
        u32 _f_41 = _fb_19[1];
        spare_free(e, cls_fit(2), _sp_19);
        _o_1 = _f_40;
        _o_2 = _f_41;
      } else if (term_aux(_f_0) == CID_ir_IAnd) {
        _o_0 = 25;
        Term _fb_20[2];
        u64 _sp_20 = ctr_take(e, _f_0, 2, _fb_20);
        u32 _f_42 = _fb_20[0];
        u32 _f_43 = _fb_20[1];
        spare_free(e, cls_fit(2), _sp_20);
        _o_1 = _f_42;
        _o_2 = _f_43;
      } else if (term_aux(_f_0) == CID_ir_IOr) {
        _o_0 = 26;
        Term _fb_21[2];
        u64 _sp_21 = ctr_take(e, _f_0, 2, _fb_21);
        u32 _f_44 = _fb_21[0];
        u32 _f_45 = _fb_21[1];
        spare_free(e, cls_fit(2), _sp_21);
        _o_1 = _f_44;
        _o_2 = _f_45;
      } else if (term_aux(_f_0) == CID_ir_INot) {
        _o_0 = 27;
        _o_1 = term_loc(_f_0);
      } else if (term_aux(_f_0) == CID_ir_IImp) {
        _o_0 = 28;
        Term _fb_22[2];
        u64 _sp_22 = ctr_take(e, _f_0, 2, _fb_22);
        u32 _f_46 = _fb_22[0];
        u32 _f_47 = _fb_22[1];
        spare_free(e, cls_fit(2), _sp_22);
        _o_1 = _f_46;
        _o_2 = _f_47;
      } else if (term_aux(_f_0) == CID_ir_IIte) {
        _o_0 = 29;
        Term _fb_23[3];
        u64 _sp_23 = ctr_take(e, _f_0, 3, _fb_23);
        u32 _f_48 = _fb_23[0];
        u32 _f_49 = _fb_23[1];
        u32 _f_50 = _fb_23[2];
        spare_free(e, cls_fit(3), _sp_23);
        _o_1 = _f_48;
        _o_2 = _f_49;
        _o_3 = _f_50;
      } else {
        _o_0 = 30;
        Term _fb_24[1];
        u64 _sp_24 = ctr_take(e, _f_0, 1, _fb_24);
        Term _f_51 = _fb_24[0];
        spare_free(e, cls_fit(1), _sp_24);
        _o_1 = _f_51;
      }
      if (_n_0 == 0) {
        term_sink(e, _f_1);
        o[0] = _o_0;
        o[1] = _o_1;
        o[2] = _o_2;
        o[3] = _o_3;
        o[4] = _o_4;
        spare_free(e, cls_fit(2), _sp_0);
      } else {
        term_sink(e, _o_1);
        term_sink(e, _o_2);
        Term _p_0 = (_n_0 - 1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_1;
        r1 = _p_0;
        _xs_0 = r0;
        _n_0 = r1;
        WL_AGAIN(spin_ir_nthNode);
      }
    }
  break;
  }
  return 1;
}

INLINE Term spin_IO_die(Env e, THR Term* o, u32 r0, Term r1, Term r2) {
  u32 wpoll = 0;
  u32 _code_0 = r0;
  Term _msg_0 = r1;
  Term _k_0 = r2;
  WL_SPIN
    term_sink(e, _k_0);
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = _code_0;
    e.mem[_nd_0 + 1] = _msg_0;
    o[0] = term_ctr(CID_Halt, _nd_0);
  break;
  }
  return 1;
}

// Work
// ====

// A host self-jump is a tail call: as a loop, clang hoists constants badly.
#if !DEVICE
#undef  WL_SPIN
#undef  WL_SPUN
#undef  WL_AGAIN
#define WL_SPIN
#define WL_SPUN
#define WL_AGAIN    WL_JMP

typedef Term (PRESERVE(preserve_none) *WlFn)(WL_SIG);
#define WL_X(F) WL_FN WL_##F(WL_SIG);
WL_TABLE WL_X(FID_ENTER)
#undef WL_X
#define WL_X(F) WL_##F,
static const WlFn wl_tab[] = { WL_TABLE };
#undef WL_X
FAR Term work_loop(Env e, DEV Term* sp, Term t, u32 seq);
#endif

static Term work_loop(Env e, DEV Term* sp, Term t, u32 seq) {
  WL_BANK
  u32 rn = 0;
  r0 = t;
#if DEVICE
  u32 fid   = FID_ENTER;
  u32 wpoll = 0;
  WL_SPIN
  switch (fid) {
#else
  return WL_FID_ENTER(WL_ALL);
}
#endif

// Segments
// ========

// A task enters through its words: a continuation's results ride r0..
// and its parameters the stack; any other segment's parameters ride r0..

  WL_CASE(FID_ir_sequenceTake)
    Term _values_0 = r0;
    Term _count_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_values_0) == CID_Con) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (_count_0 == 0) {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = term_pak(CID_Nil, 0);
        WL_RETN(1);
      } else {
        Term _rest_0 = (_count_0 - 1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(2);
          STK(0) = _f_0;
          STK(1) = FID_ir_sequenceTake_k1;
          WL_PUSHN(2);
        } else {
          u64 _t_0 = task_node(e, FID_ir_sequenceTake_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _f_0;
          WL_CONT = term_tsk(FID_ir_sequenceTake_k1, _t_0);
          WL_IDX = 1;
        }
        r0 = _f_1;
        r1 = _rest_0;
        _values_0 = r0;
        _count_0 = r1;
        WL_AGAIN(FID_ir_sequenceTake);
      }
    } else {
      if (_count_0 == 0) {
        r0 = term_pak(CID_Nil, 0);
        WL_RETN(1);
      } else {
        r0 = term_pak(CID_Nil, 0);
        WL_RETN(1);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_sequenceTake_k1)
    WL_POPN(1);
    Term _f_2 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _f_2);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_0);
    r0 = term_ctr(CID_Con, _nd_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_buildFunctionLeft)
    Term _built_0 = r0;
    Term _built_1 = r1;
    Term _built_2 = r2;
    u32 _right_size_0 = r3;
    WL_OPEN
    if (term_aux(_built_1) == CID_Con) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _built_1, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_built_2) == CID_Con) {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _built_2, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        spare_free(e, cls_fit(2), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(4);
          STK(0) = _built_0;
          STK(1) = _f_0;
          STK(2) = _f_2;
          STK(3) = FID_ir_buildFunctionLeft_k1;
          WL_PUSHN(4);
        } else {
          u64 _t_0 = task_node(e, FID_ir_buildFunctionLeft_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _built_0;
          e.mem[_t_0 + 1] = _f_0;
          e.mem[_t_0 + 2] = _f_2;
          WL_CONT = term_tsk(FID_ir_buildFunctionLeft_k1, _t_0);
          WL_IDX = 3;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_buildFunctionTree)) {
          u64 _t_1 = task_node(e, FID_ir_buildFunctionTree, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _right_size_0;
          e.mem[_t_1 + 1] = _f_1;
          e.mem[_t_1 + 2] = _f_3;
          return term_tsk(FID_ir_buildFunctionTree, _t_1);
        }
        r0 = _right_size_0;
        r1 = _f_1;
        r2 = _f_3;
        WL_JMP(FID_ir_buildFunctionTree);
      } else {
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, _f_1);
        r0 = _built_0;
        r1 = term_ctr(CID_Con, _nd_0);
        r2 = _built_2;
        WL_RETN(3);
      }
    } else {
      r0 = _built_0;
      r1 = _built_1;
      r2 = _built_2;
      WL_RETN(3);
    }
  }}

  WL_CASE(FID_ir_buildFunctionLeft_k1)
    WL_POPN(3);
    Term _built_3 = STK(0);
    Term _f_4 = STK(1);
    Term _f_5 = STK(2);
    Term _h_0 = r0;
    Term _h_1 = r1;
    Term _h_2 = r2;
    WL_OPEN
    Term _o_0[3];
    if (spin_ir_buildFunctionRight(e, _o_0, _h_0, _h_1, _h_2, _built_3, _f_4, _f_5) == 0) {
      return 0;
    }
    r0 = _o_0[0];
    r1 = _o_0[1];
    r2 = _o_0[2];
    WL_RETN(3);
  }}

  WL_CASE(FID_ir_setAppend)
    Term _values_0 = r0;
    Term _reversed_0 = r1;
    u32 _size_0 = r2;
    WL_OPEN
    WL_SPIN
    if (term_aux(_values_0) == CID_Nil) {
      Term _o_0[1];
      if (spin_List_reverse(e, _o_0, _reversed_0) == 0) {
        return 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
        u64 _t_0 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _o_0[0];
        e.mem[_t_0 + 1] = _size_0;
        return term_tsk(FID_ir_setSorted, _t_0);
      }
      r0 = _o_0[0];
      r1 = _size_0;
      WL_JMP(FID_ir_setSorted);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, _reversed_0);
      r0 = _f_1;
      r1 = term_ctr(CID_Con, _nd_0);
      r2 = U32_BIN(_size_0, +, 1);
      _values_0 = r0;
      _reversed_0 = r1;
      _size_0 = r2;
      WL_AGAIN(FID_ir_setAppend);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_functionAppendTree)
    Term _tree_0 = r0;
    Term _key_0 = r1;
    Term _value_0 = r2;
    WL_OPEN
    WL_SPIN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      Term _o_0[1];
      if (spin_ir_functionNode(e, _o_0, term_pak(CID_ir_FunctionEmpty, 0), _key_0, _value_0, term_pak(CID_ir_FunctionEmpty, 0)) == 0) {
        return 0;
      }
      r0 = _o_0[0];
      WL_RETN(1);
    } else {
      Term _fb_0[6];
      u64 _sp_0 = ctr_take(e, _tree_0, 6, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _f_2 = _fb_0[2];
      Term _f_3 = _fb_0[3];
      u32 _f_4 = _fb_0[4];
      u32 _f_5 = _fb_0[5];
      spare_free(e, cls_fit(6), _sp_0);
      if (seq) {
        WL_ROOM(4);
        STK(0) = _f_0;
        STK(1) = _f_1;
        STK(2) = _f_2;
        STK(3) = FID_ir_functionAppendTree_k1;
        WL_PUSHN(4);
      } else {
        u64 _t_0 = task_node(e, FID_ir_functionAppendTree_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        e.mem[_t_0 + 1] = _f_1;
        e.mem[_t_0 + 2] = _f_2;
        WL_CONT = term_tsk(FID_ir_functionAppendTree_k1, _t_0);
        WL_IDX = 3;
      }
      r0 = _f_3;
      r1 = _key_0;
      r2 = _value_0;
      _tree_0 = r0;
      _key_0 = r1;
      _value_0 = r2;
      WL_AGAIN(FID_ir_functionAppendTree);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_functionAppendTree_k1)
    WL_POPN(3);
    Term _f_6 = STK(0);
    Term _f_7 = STK(1);
    Term _f_8 = STK(2);
    Term _h_0 = r0;
    WL_OPEN
    Term _o_1[1];
    if (spin_ir_functionAppendBalanced(e, _o_1, _f_6, _f_7, _f_8, _h_0) == 0) {
      return 0;
    }
    r0 = _o_1[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionValues)
    Term _tree_0 = r0;
    Term _acc_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      r0 = _acc_0;
      WL_RETN(1);
    } else {
      Term _fb_0[6];
      u64 _sp_0 = ctr_take(e, _tree_0, 6, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _f_2 = _fb_0[2];
      Term _f_3 = _fb_0[3];
      u32 _f_4 = _fb_0[4];
      u32 _f_5 = _fb_0[5];
      term_sink(e, _f_1);
      spare_free(e, cls_fit(6), _sp_0);
      if (seq) {
        WL_ROOM(3);
        STK(0) = _f_0;
        STK(1) = _f_2;
        STK(2) = FID_ir_functionValues_k1;
        WL_PUSHN(3);
      } else {
        u64 _t_0 = task_node(e, FID_ir_functionValues_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        e.mem[_t_0 + 1] = _f_2;
        WL_CONT = term_tsk(FID_ir_functionValues_k1, _t_0);
        WL_IDX = 2;
      }
      r0 = _f_3;
      r1 = _acc_0;
      _tree_0 = r0;
      _acc_0 = r1;
      WL_AGAIN(FID_ir_functionValues);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_functionValues_k1)
    WL_POPN(2);
    Term _f_6 = STK(0);
    Term _f_7 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _f_7);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_0);
    if (!DEVICE && !seq && fid_nofk(FID_ir_functionValues)) {
      u64 _t_1 = task_node(e, FID_ir_functionValues, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _f_6;
      e.mem[_t_1 + 1] = term_ctr(CID_Con, _nd_0);
      return term_tsk(FID_ir_functionValues, _t_1);
    }
    r0 = _f_6;
    r1 = term_ctr(CID_Con, _nd_0);
    WL_JMP(FID_ir_functionValues);
  }}

  WL_CASE(FID_ir_sequenceValue)
    Term _values_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _values_0;
      STK(1) = FID_ir_sequenceValue_k1;
      WL_PUSHN(2);
    } else {
      u64 _t_0 = task_node(e, FID_ir_sequenceValue_k1, WL_CONT, WL_IDX, 1);
      e.mem[_t_0 + 0] = _values_0;
      WL_CONT = term_tsk(FID_ir_sequenceValue_k1, _t_0);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_vlen)) {
      u64 _t_1 = task_node(e, FID_ir_vlen, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _values_0;
      return term_tsk(FID_ir_vlen, _t_1);
    }
    r0 = _values_0;
    WL_JMP(FID_ir_vlen);
  }}

  WL_CASE(FID_ir_sequenceValue_k1)
    WL_POPN(1);
    Term _values_1 = STK(0);
    u32 _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _values_1;
      STK(1) = FID_ir_sequenceValue_k3;
      WL_PUSHN(2);
    } else {
      u64 _t_2 = task_node(e, FID_ir_sequenceValue_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _values_1;
      WL_CONT = term_tsk(FID_ir_sequenceValue_k3, _t_2);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_rangeItems)) {
      u64 _t_3 = task_node(e, FID_ir_rangeItems, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = ((u64)(_h_0));
      e.mem[_t_3 + 1] = 1ull;
      return term_tsk(FID_ir_rangeItems, _t_3);
    }
    r0 = ((u64)(_h_0));
    r1 = 1ull;
    WL_JMP(FID_ir_rangeItems);
  }}

  WL_CASE(FID_ir_sequenceValue_k3)
    WL_POPN(1);
    Term _values_2 = STK(0);
    Term _h_1 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_functionSorted)) {
      u64 _t_4 = task_node(e, FID_ir_functionSorted, WL_CONT, WL_IDX, 0);
      e.mem[_t_4 + 0] = _h_1;
      e.mem[_t_4 + 1] = _values_2;
      return term_tsk(FID_ir_functionSorted, _t_4);
    }
    r0 = _h_1;
    r1 = _values_2;
    WL_JMP(FID_ir_functionSorted);
  }}

  WL_CASE(FID_ir_domainTreeLeft)
    Term _result_0 = r0;
    Term _result_1 = r1;
    u32 _right_size_0 = r2;
    u32 _size_0 = r3;
    WL_OPEN
    if (term_aux(_result_1) == CID_Con) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _result_1, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(4);
        STK(0) = _result_0;
        STK(1) = _f_0;
        STK(2) = _size_0;
        STK(3) = FID_ir_domainTreeLeft_k1;
        WL_PUSHN(4);
      } else {
        u64 _t_0 = task_node(e, FID_ir_domainTreeLeft_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _result_0;
        e.mem[_t_0 + 1] = _f_0;
        e.mem[_t_0 + 2] = _size_0;
        WL_CONT = term_tsk(FID_ir_domainTreeLeft_k1, _t_0);
        WL_IDX = 3;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_buildDomainTree)) {
        u64 _t_1 = task_node(e, FID_ir_buildDomainTree, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _right_size_0;
        e.mem[_t_1 + 1] = _f_1;
        return term_tsk(FID_ir_buildDomainTree, _t_1);
      }
      r0 = _right_size_0;
      r1 = _f_1;
      WL_JMP(FID_ir_buildDomainTree);
    } else {
      r0 = _result_0;
      r1 = term_pak(CID_Nil, 0);
      WL_RETN(2);
    }
  }}

  WL_CASE(FID_ir_domainTreeLeft_k1)
    WL_POPN(3);
    Term _result_2 = STK(0);
    Term _f_2 = STK(1);
    u32 _size_1 = STK(2);
    Term _h_0 = r0;
    Term _h_1 = r1;
    WL_OPEN
    Term _o_0[2];
    if (spin_ir_domainTreeRight(e, _o_0, _result_2, _f_2, _size_1, _h_0, _h_1) == 0) {
      return 0;
    }
    r0 = _o_0[0];
    r1 = _o_0[1];
    WL_RETN(2);
  }}

  WL_CASE(FID_ir_productPrefix)
    Term _value_0 = r0;
    Term _rows_0 = r1;
    Term _acc_0 = r2;
    WL_OPEN
    WL_SPIN
    if (term_aux(_rows_0) == CID_Nil) {
      term_sink(e, _value_0);
      r0 = _acc_0;
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _rows_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      _value_0 = term_keep(e, _value_0, 1);
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(3);
        STK(0) = _value_0;
        STK(1) = _f_0;
        STK(2) = FID_ir_productPrefix_k1;
        WL_PUSHN(3);
      } else {
        u64 _t_0 = task_node(e, FID_ir_productPrefix_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _value_0;
        e.mem[_t_0 + 1] = _f_0;
        WL_CONT = term_tsk(FID_ir_productPrefix_k1, _t_0);
        WL_IDX = 2;
      }
      r0 = _value_0;
      r1 = _f_1;
      r2 = _acc_0;
      _value_0 = r0;
      _rows_0 = r1;
      _acc_0 = r2;
      WL_AGAIN(FID_ir_productPrefix);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_productPrefix_k1)
    WL_POPN(2);
    Term _value_1 = STK(0);
    Term _f_2 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _value_1);
    e.mem[_nd_0 + 1] = rfc_seal(e, _f_2);
    u64 _nd_1 = heap_alloc(e, 1);
    e.mem[_nd_1 + 0] = rfc_seal(e, term_ctr(CID_Con, _nd_0));
    e.mem[_nd_1 + 1] = rfc_seal(e, _h_0);
    r0 = term_ctr(CID_Con, _nd_1);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionSequenceKeys)
    Term _keys_0 = r0;
    u32 _next_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_keys_0) == CID_Nil) {
      r0 = 1;
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _keys_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VI) {
        u32 _key_0 = term_loc(_f_0);
        u32 _s_0 = U32_BIN(_key_0, ==, _next_0);
        if (_s_0 == 1) {
          u32 _next_1 = U32_BIN(_next_0, +, 1);
          spare_free(e, cls_fit(2), _sp_0);
          r0 = _f_1;
          r1 = _next_1;
          _keys_0 = r0;
          _next_0 = r1;
          WL_AGAIN(FID_ir_functionSequenceKeys);
        } else {
          term_sink(e, _f_1);
          u32 _next_2 = U32_BIN(_next_0, +, 1);
          spare_free(e, cls_fit(2), _sp_0);
          r0 = 0;
          WL_RETN(1);
        }
      } else {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = 0;
        WL_RETN(1);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_buildFunctionTree)
    u32 _size_0 = r0;
    Term _keys_0 = r1;
    Term _values_0 = r2;
    WL_OPEN
    WL_SPIN
    if (_size_0 == 0) {
      r0 = term_pak(CID_ir_FunctionEmpty, 0);
      r1 = _keys_0;
      r2 = _values_0;
      WL_RETN(3);
    } else {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _size_0;
        STK(1) = FID_ir_buildFunctionTree_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_ir_buildFunctionTree_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _size_0;
        WL_CONT = term_tsk(FID_ir_buildFunctionTree_k1, _t_0);
        WL_IDX = 1;
      }
      r0 = ((u32)(2ull) == 0 ? 0 : (u64)U32_QUO((u32)(_size_0), (u32)(2ull)));
      r1 = _keys_0;
      r2 = _values_0;
      _size_0 = r0;
      _keys_0 = r1;
      _values_0 = r2;
      WL_AGAIN(FID_ir_buildFunctionTree);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_buildFunctionTree_k1)
    WL_POPN(1);
    u32 _size_1 = STK(0);
    Term _h_0 = r0;
    Term _h_1 = r1;
    Term _h_2 = r2;
    WL_OPEN
    Term _a_0 = U32_BIN(_size_1, -, 1ull);
    Term _a_1 = ((u32)(2ull) == 0 ? 0 : (u64)U32_QUO((u32)(_size_1), (u32)(2ull)));
    if (!DEVICE && !seq && fid_nofk(FID_ir_buildFunctionLeft)) {
      u64 _t_1 = task_node(e, FID_ir_buildFunctionLeft, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _h_0;
      e.mem[_t_1 + 1] = _h_1;
      e.mem[_t_1 + 2] = _h_2;
      e.mem[_t_1 + 3] = U32_BIN(_a_0, -, _a_1);
      return term_tsk(FID_ir_buildFunctionLeft, _t_1);
    }
    r0 = _h_0;
    r1 = _h_1;
    r2 = _h_2;
    r3 = U32_BIN(_a_0, -, _a_1);
    WL_JMP(FID_ir_buildFunctionLeft);
  }}

  WL_CASE(FID_ir_powerAdd)
    Term _value_0 = r0;
    Term _subsets_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_subsets_0) == CID_Nil) {
      term_sink(e, _value_0);
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _subsets_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VSet) {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _f_0, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        _value_0 = term_keep(e, _value_0, 1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _value_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, _f_2);
        Term _o_0[1];
        if (spin_ir_domainSize(e, _o_0, _f_3) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(2), _sp_1);
        if (seq) {
          WL_ROOM(3);
          STK(0) = _value_0;
          STK(1) = _f_1;
          STK(2) = FID_ir_powerAdd_k1;
          WL_PUSHN(3);
        } else {
          u64 _t_0 = task_node(e, FID_ir_powerAdd_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _value_0;
          e.mem[_t_0 + 1] = _f_1;
          WL_CONT = term_tsk(FID_ir_powerAdd_k1, _t_0);
          WL_IDX = 2;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
          u64 _t_1 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = term_ctr(CID_Con, _nd_0);
          e.mem[_t_1 + 1] = U32_BIN((u32)_o_0[0], +, 1);
          return term_tsk(FID_ir_setSorted, _t_1);
        }
        r0 = term_ctr(CID_Con, _nd_0);
        r1 = U32_BIN((u32)_o_0[0], +, 1);
        WL_JMP(FID_ir_setSorted);
      } else {
        term_sink(e, _f_0);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_powerAdd_k5;
          WL_PUSHN(1);
        } else {
          u64 _t_4 = task_node(e, FID_ir_powerAdd_k5, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_powerAdd_k5, _t_4);
          WL_IDX = 0;
        }
        r0 = _value_0;
        r1 = _f_1;
        _value_0 = r0;
        _subsets_0 = r1;
        WL_AGAIN(FID_ir_powerAdd);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_powerAdd_k1)
    WL_POPN(2);
    Term _value_1 = STK(0);
    Term _f_4 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_0;
      STK(1) = FID_ir_powerAdd_k3;
      WL_PUSHN(2);
    } else {
      u64 _t_2 = task_node(e, FID_ir_powerAdd_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _h_0;
      WL_CONT = term_tsk(FID_ir_powerAdd_k3, _t_2);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_powerAdd)) {
      u64 _t_3 = task_node(e, FID_ir_powerAdd, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _value_1;
      e.mem[_t_3 + 1] = _f_4;
      return term_tsk(FID_ir_powerAdd, _t_3);
    }
    r0 = _value_1;
    r1 = _f_4;
    WL_JMP(FID_ir_powerAdd);
  }}

  WL_CASE(FID_ir_powerAdd_k3)
    WL_POPN(1);
    Term _h_2 = STK(0);
    Term _h_1 = r0;
    WL_OPEN
    u64 _nd_1 = heap_alloc(e, 1);
    e.mem[_nd_1 + 0] = rfc_seal(e, _h_2);
    e.mem[_nd_1 + 1] = rfc_seal(e, _h_1);
    r0 = term_ctr(CID_Con, _nd_1);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_powerAdd_k5)
    Term _h_3 = r0;
    WL_OPEN
    u64 _nd_2 = heap_alloc(e, 1);
    e.mem[_nd_2 + 0] = rfc_seal(e, term_ctr(CID_ir_VE, STAT_OFF + 295));
    e.mem[_nd_2 + 1] = rfc_seal(e, _h_3);
    r0 = term_ctr(CID_Con, _nd_2);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_setMergeLeftEmpty)
    u32 _mode_0 = r0;
    Term _right_0 = r1;
    Term _reversed_0 = r2;
    u32 _size_0 = r3;
    WL_OPEN
    if (_mode_0 == 1) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setAppend)) {
        u64 _t_0 = task_node(e, FID_ir_setAppend, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _right_0;
        e.mem[_t_0 + 1] = _reversed_0;
        e.mem[_t_0 + 2] = _size_0;
        return term_tsk(FID_ir_setAppend, _t_0);
      }
      r0 = _right_0;
      r1 = _reversed_0;
      r2 = _size_0;
      WL_JMP(FID_ir_setAppend);
    } else {
      term_sink(e, _right_0);
      Term _o_0[1];
      if (spin_List_reverse(e, _o_0, _reversed_0) == 0) {
        return 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
        u64 _t_1 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _o_0[0];
        e.mem[_t_1 + 1] = _size_0;
        return term_tsk(FID_ir_setSorted, _t_1);
      }
      r0 = _o_0[0];
      r1 = _size_0;
      WL_JMP(FID_ir_setSorted);
    }
  }}

  WL_CASE(FID_List_merge_go_0)
    Term _fuel_0 = r0;
    Term _st_0 = r1;
    Term _st_1 = r2;
    Term _st_2 = r3;
    WL_OPEN
    if (_fuel_0 == 0) {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _st_0;
        STK(1) = FID_List_merge_go_0_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_List_merge_go_0_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _st_0;
        WL_CONT = term_tsk(FID_List_merge_go_0_k1, _t_0);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_List_append)) {
        u64 _t_1 = task_node(e, FID_List_append, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _st_1;
        e.mem[_t_1 + 1] = _st_2;
        return term_tsk(FID_List_append, _t_1);
      }
      r0 = _st_1;
      r1 = _st_2;
      WL_JMP(FID_List_append);
    } else {
      Term _f_0 = (_fuel_0 - 1);
      if (term_aux(_st_1) == CID_Nil) {
        if (term_aux(_st_2) == CID_Nil) {
          Term _o_1[1];
          if (spin_List_reverse_go(e, _o_1, _st_0, term_pak(CID_Nil, 0)) == 0) {
            return 0;
          }
          r0 = _o_1[0];
          WL_RETN(1);
        } else {
          Term _fb_0[2];
          u64 _sp_0 = ctr_take(e, _st_2, 2, _fb_0);
          Term _f_1 = _fb_0[0];
          Term _f_2 = _fb_0[1];
          u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
          e.mem[_nd_0 + 0] = rfc_seal(e, _f_1);
          e.mem[_nd_0 + 1] = rfc_seal(e, _f_2);
          Term _o_2[1];
          if (spin_List_reverse_go(e, _o_2, _st_0, term_ctr(CID_Con, _nd_0)) == 0) {
            return 0;
          }
          r0 = _o_2[0];
          WL_RETN(1);
        }
      } else {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _st_1, 2, _fb_1);
        Term _f_3 = _fb_1[0];
        Term _f_4 = _fb_1[1];
        if (term_aux(_st_2) == CID_Nil) {
          u64 _nd_1 = _sp_1 >= HEAP_OFF ? _sp_1 : heap_alloc(e, 1);
          e.mem[_nd_1 + 0] = rfc_seal(e, _f_3);
          e.mem[_nd_1 + 1] = rfc_seal(e, _f_4);
          Term _o_3[1];
          if (spin_List_reverse_go(e, _o_3, _st_0, term_ctr(CID_Con, _nd_1)) == 0) {
            return 0;
          }
          r0 = _o_3[0];
          WL_RETN(1);
        } else {
          Term _fb_2[2];
          u64 _sp_2 = ctr_take(e, _st_2, 2, _fb_2);
          Term _f_5 = _fb_2[0];
          Term _f_6 = _fb_2[1];
          _f_3 = term_keep(e, _f_3, 1);
          _f_5 = term_keep(e, _f_5, 1);
          spare_free(e, cls_fit(2), _sp_2);
          spare_free(e, cls_fit(2), _sp_1);
          if (seq) {
            WL_ROOM(7);
            STK(0) = _f_0;
            STK(1) = _st_0;
            STK(2) = _f_3;
            STK(3) = _f_4;
            STK(4) = _f_5;
            STK(5) = _f_6;
            STK(6) = FID_List_merge_go_0_k3;
            WL_PUSHN(7);
          } else {
            u64 _t_2 = task_node(e, FID_List_merge_go_0_k3, WL_CONT, WL_IDX, 1);
            e.mem[_t_2 + 0] = _f_0;
            e.mem[_t_2 + 1] = _st_0;
            e.mem[_t_2 + 2] = _f_3;
            e.mem[_t_2 + 3] = _f_4;
            e.mem[_t_2 + 4] = _f_5;
            e.mem[_t_2 + 5] = _f_6;
            WL_CONT = term_tsk(FID_List_merge_go_0_k3, _t_2);
            WL_IDX = 6;
          }
          if (!DEVICE && !seq && fid_nofk(FID_ir_valueLe)) {
            u64 _t_3 = task_node(e, FID_ir_valueLe, WL_CONT, WL_IDX, 0);
            e.mem[_t_3 + 0] = _f_3;
            e.mem[_t_3 + 1] = _f_5;
            return term_tsk(FID_ir_valueLe, _t_3);
          }
          r0 = _f_3;
          r1 = _f_5;
          WL_JMP(FID_ir_valueLe);
        }
      }
    }
  }}

  WL_CASE(FID_List_merge_go_0_k1)
    WL_POPN(1);
    Term _st_3 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    Term _o_0[1];
    if (spin_List_reverse_go(e, _o_0, _st_3, _h_0) == 0) {
      return 0;
    }
    r0 = _o_0[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_List_merge_go_0_k3)
    WL_POPN(6);
    Term _f_7 = STK(0);
    Term _st_4 = STK(1);
    Term _f_8 = STK(2);
    Term _f_9 = STK(3);
    Term _f_10 = STK(4);
    Term _f_11 = STK(5);
    u32 _h_1 = r0;
    WL_OPEN
    Term _o_4[3];
    if (spin_List_merge_step(e, _o_4, _st_4, _f_8, _f_9, _f_10, _f_11, _h_1) == 0) {
      return 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_List_merge_go_0)) {
      u64 _t_4 = task_node(e, FID_List_merge_go_0, WL_CONT, WL_IDX, 0);
      e.mem[_t_4 + 0] = _f_7;
      e.mem[_t_4 + 1] = _o_4[0];
      e.mem[_t_4 + 2] = _o_4[1];
      e.mem[_t_4 + 3] = _o_4[2];
      return term_tsk(FID_List_merge_go_0, _t_4);
    }
    r0 = _f_7;
    r1 = _o_4[0];
    r2 = _o_4[1];
    r3 = _o_4[2];
    WL_JMP(FID_List_merge_go_0);
  }}

  WL_CASE(FID_List_append)
    Term _xs_0 = r0;
    Term _ys_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_xs_0) == CID_Nil) {
      r0 = _ys_0;
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_0;
        STK(1) = FID_List_append_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_List_append_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        WL_CONT = term_tsk(FID_List_append_k1, _t_0);
        WL_IDX = 1;
      }
      r0 = _f_1;
      r1 = _ys_0;
      _xs_0 = r0;
      _ys_0 = r1;
      WL_AGAIN(FID_List_append);
    }
    WL_SPUN
  }}

  WL_CASE(FID_List_append_k1)
    WL_POPN(1);
    Term _f_2 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _f_2);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_0);
    r0 = term_ctr(CID_Con, _nd_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionOrderStacks)
    Term _xs_0 = r0;
    Term _ys_0 = r1;
    WL_OPEN
    if (term_aux(_xs_0) == CID_Nil) {
      if (term_aux(_ys_0) == CID_Nil) {
        r0 = 1;
        WL_RETN(1);
      } else {
        term_sink(e, _ys_0);
        r0 = 0;
        WL_RETN(1);
      }
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_FunctionNode) {
        Term _fb_1[6];
        u64 _sp_1 = ctr_take(e, _f_0, 6, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        Term _f_4 = _fb_1[2];
        Term _f_5 = _fb_1[3];
        u32 _f_6 = _fb_1[4];
        u32 _f_7 = _fb_1[5];
        term_sink(e, _f_2);
        if (term_aux(_ys_0) == CID_Con) {
          Term _fb_2[2];
          u64 _sp_2 = ctr_take(e, _ys_0, 2, _fb_2);
          Term _f_8 = _fb_2[0];
          Term _f_9 = _fb_2[1];
          if (term_aux(_f_8) == CID_ir_FunctionNode) {
            Term _fb_3[6];
            u64 _sp_3 = ctr_take(e, _f_8, 6, _fb_3);
            Term _f_10 = _fb_3[0];
            Term _f_11 = _fb_3[1];
            Term _f_12 = _fb_3[2];
            Term _f_13 = _fb_3[3];
            u32 _f_14 = _fb_3[4];
            u32 _f_15 = _fb_3[5];
            term_sink(e, _f_10);
            spare_free(e, cls_fit(6), _sp_3);
            spare_free(e, cls_fit(2), _sp_2);
            spare_free(e, cls_fit(6), _sp_1);
            spare_free(e, cls_fit(2), _sp_0);
            if (seq) {
              WL_ROOM(7);
              STK(0) = _f_4;
              STK(1) = _f_5;
              STK(2) = _f_1;
              STK(3) = _f_12;
              STK(4) = _f_13;
              STK(5) = _f_9;
              STK(6) = FID_ir_functionOrderStacks_k1;
              WL_PUSHN(7);
            } else {
              u64 _t_0 = task_node(e, FID_ir_functionOrderStacks_k1, WL_CONT, WL_IDX, 1);
              e.mem[_t_0 + 0] = _f_4;
              e.mem[_t_0 + 1] = _f_5;
              e.mem[_t_0 + 2] = _f_1;
              e.mem[_t_0 + 3] = _f_12;
              e.mem[_t_0 + 4] = _f_13;
              e.mem[_t_0 + 5] = _f_9;
              WL_CONT = term_tsk(FID_ir_functionOrderStacks_k1, _t_0);
              WL_IDX = 6;
            }
            if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
              u64 _t_1 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
              e.mem[_t_1 + 0] = _f_3;
              e.mem[_t_1 + 1] = _f_11;
              return term_tsk(FID_ir_valueOrder, _t_1);
            }
            r0 = _f_3;
            r1 = _f_11;
            WL_JMP(FID_ir_valueOrder);
          } else {
            term_sink(e, _f_3);
            term_sink(e, _f_4);
            term_sink(e, _f_5);
            term_sink(e, _f_1);
            term_sink(e, _f_8);
            term_sink(e, _f_9);
            spare_free(e, cls_fit(2), _sp_2);
            spare_free(e, cls_fit(6), _sp_1);
            spare_free(e, cls_fit(2), _sp_0);
            r0 = 1;
            WL_RETN(1);
          }
        } else {
          term_sink(e, _f_3);
          term_sink(e, _f_4);
          term_sink(e, _f_5);
          term_sink(e, _f_1);
          spare_free(e, cls_fit(6), _sp_1);
          spare_free(e, cls_fit(2), _sp_0);
          r0 = 2;
          WL_RETN(1);
        }
      } else {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        if (term_aux(_ys_0) == CID_Nil) {
          spare_free(e, cls_fit(2), _sp_0);
          r0 = 2;
          WL_RETN(1);
        } else {
          term_sink(e, _ys_0);
          spare_free(e, cls_fit(2), _sp_0);
          r0 = 1;
          WL_RETN(1);
        }
      }
    }
  }}

  WL_CASE(FID_ir_functionOrderStacks_k1)
    WL_POPN(6);
    Term _f_16 = STK(0);
    Term _f_17 = STK(1);
    Term _f_18 = STK(2);
    Term _f_19 = STK(3);
    Term _f_20 = STK(4);
    Term _f_21 = STK(5);
    u32 _h_0 = r0;
    WL_OPEN
    if (_h_0 == 1) {
      if (seq) {
        WL_ROOM(5);
        STK(0) = _f_17;
        STK(1) = _f_18;
        STK(2) = _f_20;
        STK(3) = _f_21;
        STK(4) = FID_ir_functionOrderStacks_k3;
        WL_PUSHN(5);
      } else {
        u64 _t_2 = task_node(e, FID_ir_functionOrderStacks_k3, WL_CONT, WL_IDX, 1);
        e.mem[_t_2 + 0] = _f_17;
        e.mem[_t_2 + 1] = _f_18;
        e.mem[_t_2 + 2] = _f_20;
        e.mem[_t_2 + 3] = _f_21;
        WL_CONT = term_tsk(FID_ir_functionOrderStacks_k3, _t_2);
        WL_IDX = 4;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
        u64 _t_3 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _f_16;
        e.mem[_t_3 + 1] = _f_19;
        return term_tsk(FID_ir_valueOrder, _t_3);
      }
      r0 = _f_16;
      r1 = _f_19;
      WL_JMP(FID_ir_valueOrder);
    } else {
      term_sink(e, _f_16);
      term_sink(e, _f_19);
      term_sink(e, _f_17);
      term_sink(e, _f_18);
      term_sink(e, _f_20);
      term_sink(e, _f_21);
      r0 = _h_0;
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_functionOrderStacks_k3)
    WL_POPN(4);
    Term _f_22 = STK(0);
    Term _f_23 = STK(1);
    Term _f_24 = STK(2);
    Term _f_25 = STK(3);
    u32 _h_1 = r0;
    WL_OPEN
    if (_h_1 == 1) {
      Term _o_0[1];
      if (spin_ir_functionStack(e, _o_0, _f_22, _f_23) == 0) {
        return 0;
      }
      Term _o_1[1];
      if (spin_ir_functionStack(e, _o_1, _f_24, _f_25) == 0) {
        return 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionOrderStacks)) {
        u64 _t_4 = task_node(e, FID_ir_functionOrderStacks, WL_CONT, WL_IDX, 0);
        e.mem[_t_4 + 0] = _o_0[0];
        e.mem[_t_4 + 1] = _o_1[0];
        return term_tsk(FID_ir_functionOrderStacks, _t_4);
      }
      r0 = _o_0[0];
      r1 = _o_1[0];
      WL_JMP(FID_ir_functionOrderStacks);
    } else {
      term_sink(e, _f_22);
      term_sink(e, _f_23);
      term_sink(e, _f_24);
      term_sink(e, _f_25);
      r0 = _h_1;
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_buildDomainTree)
    u32 _size_0 = r0;
    Term _values_0 = r1;
    WL_OPEN
    WL_SPIN
    if (_size_0 == 0) {
      r0 = term_pak(CID_ir_DomainEmpty, 0);
      r1 = _values_0;
      WL_RETN(2);
    } else {
      u32 _half_0 = ((u32)(2ull) == 0 ? 0 : (u64)U32_QUO((u32)(_size_0), (u32)(2ull)));
      if (seq) {
        WL_ROOM(3);
        STK(0) = _size_0;
        STK(1) = _half_0;
        STK(2) = FID_ir_buildDomainTree_k1;
        WL_PUSHN(3);
      } else {
        u64 _t_0 = task_node(e, FID_ir_buildDomainTree_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _size_0;
        e.mem[_t_0 + 1] = _half_0;
        WL_CONT = term_tsk(FID_ir_buildDomainTree_k1, _t_0);
        WL_IDX = 2;
      }
      r0 = _half_0;
      r1 = _values_0;
      _size_0 = r0;
      _values_0 = r1;
      WL_AGAIN(FID_ir_buildDomainTree);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_buildDomainTree_k1)
    WL_POPN(2);
    u32 _size_1 = STK(0);
    u32 _half_1 = STK(1);
    Term _h_0 = r0;
    Term _h_1 = r1;
    WL_OPEN
    Term _a_0 = U32_BIN(_size_1, -, _half_1);
    if (!DEVICE && !seq && fid_nofk(FID_ir_domainTreeLeft)) {
      u64 _t_1 = task_node(e, FID_ir_domainTreeLeft, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _h_0;
      e.mem[_t_1 + 1] = _h_1;
      e.mem[_t_1 + 2] = U32_BIN(_a_0, -, 1ull);
      e.mem[_t_1 + 3] = _size_1;
      return term_tsk(FID_ir_domainTreeLeft, _t_1);
    }
    r0 = _h_0;
    r1 = _h_1;
    r2 = U32_BIN(_a_0, -, 1ull);
    r3 = _size_1;
    WL_JMP(FID_ir_domainTreeLeft);
  }}

  WL_CASE(FID_ir_integerSequence)
    u32 _previous_0 = r0;
    Term _values_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_values_0) == CID_Nil) {
      r0 = 1;
      WL_RETN(1);
    } else {
      u64 _sp_0 = term_peek(e.mem, _values_0);
      Term _f_0 = e.mem[_sp_0 + 0];
      Term _f_1 = e.mem[_sp_0 + 1];
      if (term_aux(_f_0) == CID_ir_VI) {
        u32 _value_0 = term_loc(_f_0);
        Term _a_0 = U32_BIN(_previous_0, +, 1);
        Term _o_0[1];
        if (spin_Bool_and(e, _o_0, U32_BIN(_previous_0, !=, 2147483647ull), U32_BIN(_value_0, ==, _a_0)) == 0) {
          return 0;
        }
        if ((u32)_o_0[0] == 1) {
          r0 = _value_0;
          r1 = _f_1;
          _previous_0 = r0;
          _values_0 = r1;
          WL_AGAIN(FID_ir_integerSequence);
        } else {
          r0 = 0;
          WL_RETN(1);
        }
      } else {
        r0 = 0;
        WL_RETN(1);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_productColumn)
    Term _values_0 = r0;
    Term _rows_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_values_0) == CID_Nil) {
      term_sink(e, _rows_0);
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      _rows_0 = term_keep(e, _rows_0, 1);
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(3);
        STK(0) = _f_0;
        STK(1) = _rows_0;
        STK(2) = FID_ir_productColumn_k1;
        WL_PUSHN(3);
      } else {
        u64 _t_0 = task_node(e, FID_ir_productColumn_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        e.mem[_t_0 + 1] = _rows_0;
        WL_CONT = term_tsk(FID_ir_productColumn_k1, _t_0);
        WL_IDX = 2;
      }
      r0 = _f_1;
      r1 = _rows_0;
      _values_0 = r0;
      _rows_0 = r1;
      WL_AGAIN(FID_ir_productColumn);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_productColumn_k1)
    WL_POPN(2);
    Term _f_2 = STK(0);
    Term _rows_1 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_productPrefix)) {
      u64 _t_1 = task_node(e, FID_ir_productPrefix, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _f_2;
      e.mem[_t_1 + 1] = _rows_1;
      e.mem[_t_1 + 2] = _h_0;
      return term_tsk(FID_ir_productPrefix, _t_1);
    }
    r0 = _f_2;
    r1 = _rows_1;
    r2 = _h_0;
    WL_JMP(FID_ir_productPrefix);
  }}

  WL_CASE(FID_ir_boundedBoolean)
    Term _nodes_0 = r0;
    Term _env_0 = r1;
    Term _env_1 = r2;
    u32 _op_0 = r3;
    u32 _slot_0 = r4;
    u32 _body_0 = r5;
    Term _original_0 = r6;
    Term _rest_0 = r7;
    Term _acc_0 = r8;
    Term _value_0 = r9;
    WL_OPEN
    if (term_aux(_value_0) == CID_ir_VB) {
      u32 _truth_0 = term_loc(_value_0);
      if (_truth_0 == 0) {
        if (_op_0 == 1) {
          term_sink(e, _nodes_0);
          term_sink(e, _env_0);
          term_sink(e, _env_1);
          term_sink(e, _original_0);
          term_sink(e, _rest_0);
          term_sink(e, _acc_0);
          r0 = term_pak(CID_ir_VB, 0);
          WL_RETN(1);
        } else {
          term_sink(e, _original_0);
          if (!DEVICE && !seq && fid_nofk(FID_ir_boundedLoop)) {
            u64 _t_0 = task_node(e, FID_ir_boundedLoop, WL_CONT, WL_IDX, 0);
            e.mem[_t_0 + 0] = _nodes_0;
            e.mem[_t_0 + 1] = _env_0;
            e.mem[_t_0 + 2] = _env_1;
            e.mem[_t_0 + 3] = _op_0;
            e.mem[_t_0 + 4] = _slot_0;
            e.mem[_t_0 + 5] = _body_0;
            e.mem[_t_0 + 6] = _rest_0;
            e.mem[_t_0 + 7] = _acc_0;
            return term_tsk(FID_ir_boundedLoop, _t_0);
          }
          r0 = _nodes_0;
          r1 = _env_0;
          r2 = _env_1;
          r3 = _op_0;
          r4 = _slot_0;
          r5 = _body_0;
          r6 = _rest_0;
          r7 = _acc_0;
          WL_JMP(FID_ir_boundedLoop);
        }
      } else {
        if (_op_0 == 2) {
          term_sink(e, _nodes_0);
          term_sink(e, _env_0);
          term_sink(e, _env_1);
          term_sink(e, _original_0);
          term_sink(e, _rest_0);
          term_sink(e, _acc_0);
          r0 = term_pak(CID_ir_VB, 1);
          WL_RETN(1);
        } else if ((_op_0 & 1) == 0) {
          term_sink(e, _original_0);
          if (!DEVICE && !seq && fid_nofk(FID_ir_boundedLoop)) {
            u64 _t_1 = task_node(e, FID_ir_boundedLoop, WL_CONT, WL_IDX, 0);
            e.mem[_t_1 + 0] = _nodes_0;
            e.mem[_t_1 + 1] = _env_0;
            e.mem[_t_1 + 2] = _env_1;
            e.mem[_t_1 + 3] = _op_0;
            e.mem[_t_1 + 4] = _slot_0;
            e.mem[_t_1 + 5] = _body_0;
            e.mem[_t_1 + 6] = _rest_0;
            e.mem[_t_1 + 7] = _acc_0;
            return term_tsk(FID_ir_boundedLoop, _t_1);
          }
          r0 = _nodes_0;
          r1 = _env_0;
          r2 = _env_1;
          r3 = _op_0;
          r4 = _slot_0;
          r5 = _body_0;
          r6 = _rest_0;
          r7 = _acc_0;
          WL_JMP(FID_ir_boundedLoop);
        } else if (_op_0 == 3) {
          u64 _nd_0 = heap_alloc(e, 1);
          e.mem[_nd_0 + 0] = rfc_seal(e, _original_0);
          e.mem[_nd_0 + 1] = rfc_seal(e, _acc_0);
          if (!DEVICE && !seq && fid_nofk(FID_ir_boundedLoop)) {
            u64 _t_2 = task_node(e, FID_ir_boundedLoop, WL_CONT, WL_IDX, 0);
            e.mem[_t_2 + 0] = _nodes_0;
            e.mem[_t_2 + 1] = _env_0;
            e.mem[_t_2 + 2] = _env_1;
            e.mem[_t_2 + 3] = 3ull;
            e.mem[_t_2 + 4] = _slot_0;
            e.mem[_t_2 + 5] = _body_0;
            e.mem[_t_2 + 6] = _rest_0;
            e.mem[_t_2 + 7] = term_ctr(CID_Con, _nd_0);
            return term_tsk(FID_ir_boundedLoop, _t_2);
          }
          r0 = _nodes_0;
          r1 = _env_0;
          r2 = _env_1;
          r3 = 3ull;
          r4 = _slot_0;
          r5 = _body_0;
          r6 = _rest_0;
          r7 = term_ctr(CID_Con, _nd_0);
          WL_JMP(FID_ir_boundedLoop);
        } else {
          term_sink(e, _original_0);
          if (!DEVICE && !seq && fid_nofk(FID_ir_boundedLoop)) {
            u64 _t_3 = task_node(e, FID_ir_boundedLoop, WL_CONT, WL_IDX, 0);
            e.mem[_t_3 + 0] = _nodes_0;
            e.mem[_t_3 + 1] = _env_0;
            e.mem[_t_3 + 2] = _env_1;
            e.mem[_t_3 + 3] = _op_0;
            e.mem[_t_3 + 4] = _slot_0;
            e.mem[_t_3 + 5] = _body_0;
            e.mem[_t_3 + 6] = _rest_0;
            e.mem[_t_3 + 7] = _acc_0;
            return term_tsk(FID_ir_boundedLoop, _t_3);
          }
          r0 = _nodes_0;
          r1 = _env_0;
          r2 = _env_1;
          r3 = _op_0;
          r4 = _slot_0;
          r5 = _body_0;
          r6 = _rest_0;
          r7 = _acc_0;
          WL_JMP(FID_ir_boundedLoop);
        }
      }
    } else {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      term_sink(e, _original_0);
      term_sink(e, _rest_0);
      term_sink(e, _acc_0);
      term_sink(e, _value_0);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 458);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_rangeItems)
    Term _fuel_0 = r0;
    u32 _first_0 = r1;
    WL_OPEN
    WL_SPIN
    if (_fuel_0 == 0) {
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _rest_0 = (_fuel_0 - 1);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _first_0;
        STK(1) = FID_ir_rangeItems_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_ir_rangeItems_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _first_0;
        WL_CONT = term_tsk(FID_ir_rangeItems_k1, _t_0);
        WL_IDX = 1;
      }
      r0 = _rest_0;
      r1 = U32_BIN(_first_0, +, 1);
      _fuel_0 = r0;
      _first_0 = r1;
      WL_AGAIN(FID_ir_rangeItems);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_rangeItems_k1)
    WL_POPN(1);
    u32 _first_1 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, term_pak(CID_ir_VI, _first_1));
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_0);
    r0 = term_ctr(CID_Con, _nd_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_setSubset)
    Term _left_0 = r0;
    Term _right_0 = r1;
    WL_OPEN
    if (term_aux(_left_0) == CID_Nil) {
      if (term_aux(_right_0) == CID_Nil) {
        r0 = 1;
        WL_RETN(1);
      } else {
        term_sink(e, _right_0);
        r0 = 1;
        WL_RETN(1);
      }
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _left_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_right_0) == CID_Nil) {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = 0;
        WL_RETN(1);
      } else {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _right_0, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        _f_0 = term_keep(e, _f_0, 1);
        spare_free(e, cls_fit(2), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(4);
          STK(0) = _f_0;
          STK(1) = _f_1;
          STK(2) = _f_3;
          STK(3) = FID_ir_setSubset_k1;
          WL_PUSHN(4);
        } else {
          u64 _t_0 = task_node(e, FID_ir_setSubset_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _f_0;
          e.mem[_t_0 + 1] = _f_1;
          e.mem[_t_0 + 2] = _f_3;
          WL_CONT = term_tsk(FID_ir_setSubset_k1, _t_0);
          WL_IDX = 3;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
          u64 _t_1 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _f_0;
          e.mem[_t_1 + 1] = _f_2;
          return term_tsk(FID_ir_valueOrder, _t_1);
        }
        r0 = _f_0;
        r1 = _f_2;
        WL_JMP(FID_ir_valueOrder);
      }
    }
  }}

  WL_CASE(FID_ir_setSubset_k1)
    WL_POPN(3);
    Term _f_4 = STK(0);
    Term _f_5 = STK(1);
    Term _f_6 = STK(2);
    u32 _h_0 = r0;
    WL_OPEN
    if (_h_0 == 0) {
      term_sink(e, _f_4);
      term_sink(e, _f_5);
      term_sink(e, _f_6);
      r0 = 0;
      WL_RETN(1);
    } else if (_h_0 == 1) {
      term_sink(e, _f_4);
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSubset)) {
        u64 _t_2 = task_node(e, FID_ir_setSubset, WL_CONT, WL_IDX, 0);
        e.mem[_t_2 + 0] = _f_5;
        e.mem[_t_2 + 1] = _f_6;
        return term_tsk(FID_ir_setSubset, _t_2);
      }
      r0 = _f_5;
      r1 = _f_6;
      WL_JMP(FID_ir_setSubset);
    } else {
      u64 _nd_0 = heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_4);
      e.mem[_nd_0 + 1] = rfc_seal(e, _f_5);
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSubset)) {
        u64 _t_3 = task_node(e, FID_ir_setSubset, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = term_ctr(CID_Con, _nd_0);
        e.mem[_t_3 + 1] = _f_6;
        return term_tsk(FID_ir_setSubset, _t_3);
      }
      r0 = term_ctr(CID_Con, _nd_0);
      r1 = _f_6;
      WL_JMP(FID_ir_setSubset);
    }
  }}

  WL_CASE(FID_ir_setMerge)
    u32 _mode_0 = r0;
    Term _left_0 = r1;
    Term _right_0 = r2;
    Term _reversed_0 = r3;
    u32 _size_0 = r4;
    WL_OPEN
    if (term_aux(_left_0) == CID_Nil) {
      if (term_aux(_right_0) == CID_Nil) {
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMergeLeftEmpty)) {
          u64 _t_0 = task_node(e, FID_ir_setMergeLeftEmpty, WL_CONT, WL_IDX, 0);
          e.mem[_t_0 + 0] = _mode_0;
          e.mem[_t_0 + 1] = term_pak(CID_Nil, 0);
          e.mem[_t_0 + 2] = _reversed_0;
          e.mem[_t_0 + 3] = _size_0;
          return term_tsk(FID_ir_setMergeLeftEmpty, _t_0);
        }
        r0 = _mode_0;
        r1 = term_pak(CID_Nil, 0);
        r2 = _reversed_0;
        r3 = _size_0;
        WL_JMP(FID_ir_setMergeLeftEmpty);
      } else {
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMergeLeftEmpty)) {
          u64 _t_1 = task_node(e, FID_ir_setMergeLeftEmpty, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _mode_0;
          e.mem[_t_1 + 1] = _right_0;
          e.mem[_t_1 + 2] = _reversed_0;
          e.mem[_t_1 + 3] = _size_0;
          return term_tsk(FID_ir_setMergeLeftEmpty, _t_1);
        }
        r0 = _mode_0;
        r1 = _right_0;
        r2 = _reversed_0;
        r3 = _size_0;
        WL_JMP(FID_ir_setMergeLeftEmpty);
      }
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _left_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_right_0) == CID_Nil) {
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, _f_1);
        if (_mode_0 == 2) {
          Term _left_1 = term_ctr(CID_Con, _nd_0);
          term_sink(e, _left_1);
          Term _o_0[1];
          if (spin_List_reverse(e, _o_0, _reversed_0) == 0) {
            return 0;
          }
          if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
            u64 _t_2 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
            e.mem[_t_2 + 0] = _o_0[0];
            e.mem[_t_2 + 1] = _size_0;
            return term_tsk(FID_ir_setSorted, _t_2);
          }
          r0 = _o_0[0];
          r1 = _size_0;
          WL_JMP(FID_ir_setSorted);
        } else {
          Term _left_2 = term_ctr(CID_Con, _nd_0);
          if (!DEVICE && !seq && fid_nofk(FID_ir_setAppend)) {
            u64 _t_3 = task_node(e, FID_ir_setAppend, WL_CONT, WL_IDX, 0);
            e.mem[_t_3 + 0] = _left_2;
            e.mem[_t_3 + 1] = _reversed_0;
            e.mem[_t_3 + 2] = _size_0;
            return term_tsk(FID_ir_setAppend, _t_3);
          }
          r0 = _left_2;
          r1 = _reversed_0;
          r2 = _size_0;
          WL_JMP(FID_ir_setAppend);
        }
      } else {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _right_0, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        _f_0 = term_keep(e, _f_0, 1);
        _f_2 = term_keep(e, _f_2, 1);
        spare_free(e, cls_fit(2), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(8);
          STK(0) = _mode_0;
          STK(1) = _f_0;
          STK(2) = _f_1;
          STK(3) = _f_2;
          STK(4) = _f_3;
          STK(5) = _reversed_0;
          STK(6) = _size_0;
          STK(7) = FID_ir_setMerge_k1;
          WL_PUSHN(8);
        } else {
          u64 _t_4 = task_node(e, FID_ir_setMerge_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_4 + 0] = _mode_0;
          e.mem[_t_4 + 1] = _f_0;
          e.mem[_t_4 + 2] = _f_1;
          e.mem[_t_4 + 3] = _f_2;
          e.mem[_t_4 + 4] = _f_3;
          e.mem[_t_4 + 5] = _reversed_0;
          e.mem[_t_4 + 6] = _size_0;
          WL_CONT = term_tsk(FID_ir_setMerge_k1, _t_4);
          WL_IDX = 7;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
          u64 _t_5 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
          e.mem[_t_5 + 0] = _f_0;
          e.mem[_t_5 + 1] = _f_2;
          return term_tsk(FID_ir_valueOrder, _t_5);
        }
        r0 = _f_0;
        r1 = _f_2;
        WL_JMP(FID_ir_valueOrder);
      }
    }
  }}

  WL_CASE(FID_ir_setMerge_k1)
    WL_POPN(7);
    u32 _mode_1 = STK(0);
    Term _f_4 = STK(1);
    Term _f_5 = STK(2);
    Term _f_6 = STK(3);
    Term _f_7 = STK(4);
    Term _reversed_1 = STK(5);
    u32 _size_1 = STK(6);
    u32 _h_0 = r0;
    WL_OPEN
    if (_h_0 == 0) {
      u64 _nd_1 = heap_alloc(e, 1);
      e.mem[_nd_1 + 0] = rfc_seal(e, _f_6);
      e.mem[_nd_1 + 1] = rfc_seal(e, _f_7);
      if (_mode_1 == 2) {
        term_sink(e, _f_4);
        Term _right_1 = term_ctr(CID_Con, _nd_1);
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
          u64 _t_6 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
          e.mem[_t_6 + 0] = 2ull;
          e.mem[_t_6 + 1] = _f_5;
          e.mem[_t_6 + 2] = _right_1;
          e.mem[_t_6 + 3] = _reversed_1;
          e.mem[_t_6 + 4] = _size_1;
          return term_tsk(FID_ir_setMerge, _t_6);
        }
        r0 = 2ull;
        r1 = _f_5;
        r2 = _right_1;
        r3 = _reversed_1;
        r4 = _size_1;
        WL_JMP(FID_ir_setMerge);
      } else {
        Term _right_2 = term_ctr(CID_Con, _nd_1);
        u64 _nd_2 = heap_alloc(e, 1);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_4);
        e.mem[_nd_2 + 1] = rfc_seal(e, _reversed_1);
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
          u64 _t_7 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
          e.mem[_t_7 + 0] = _mode_1;
          e.mem[_t_7 + 1] = _f_5;
          e.mem[_t_7 + 2] = _right_2;
          e.mem[_t_7 + 3] = term_ctr(CID_Con, _nd_2);
          e.mem[_t_7 + 4] = U32_BIN(_size_1, +, 1);
          return term_tsk(FID_ir_setMerge, _t_7);
        }
        r0 = _mode_1;
        r1 = _f_5;
        r2 = _right_2;
        r3 = term_ctr(CID_Con, _nd_2);
        r4 = U32_BIN(_size_1, +, 1);
        WL_JMP(FID_ir_setMerge);
      }
    } else if (_h_0 == 1) {
      term_sink(e, _f_6);
      if (_mode_1 == 3) {
        term_sink(e, _f_4);
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
          u64 _t_8 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
          e.mem[_t_8 + 0] = 3ull;
          e.mem[_t_8 + 1] = _f_5;
          e.mem[_t_8 + 2] = _f_7;
          e.mem[_t_8 + 3] = _reversed_1;
          e.mem[_t_8 + 4] = _size_1;
          return term_tsk(FID_ir_setMerge, _t_8);
        }
        r0 = 3ull;
        r1 = _f_5;
        r2 = _f_7;
        r3 = _reversed_1;
        r4 = _size_1;
        WL_JMP(FID_ir_setMerge);
      } else {
        u64 _nd_3 = heap_alloc(e, 1);
        e.mem[_nd_3 + 0] = rfc_seal(e, _f_4);
        e.mem[_nd_3 + 1] = rfc_seal(e, _reversed_1);
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
          u64 _t_9 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
          e.mem[_t_9 + 0] = _mode_1;
          e.mem[_t_9 + 1] = _f_5;
          e.mem[_t_9 + 2] = _f_7;
          e.mem[_t_9 + 3] = term_ctr(CID_Con, _nd_3);
          e.mem[_t_9 + 4] = U32_BIN(_size_1, +, 1);
          return term_tsk(FID_ir_setMerge, _t_9);
        }
        r0 = _mode_1;
        r1 = _f_5;
        r2 = _f_7;
        r3 = term_ctr(CID_Con, _nd_3);
        r4 = U32_BIN(_size_1, +, 1);
        WL_JMP(FID_ir_setMerge);
      }
    } else {
      u64 _nd_4 = heap_alloc(e, 1);
      e.mem[_nd_4 + 0] = rfc_seal(e, _f_4);
      e.mem[_nd_4 + 1] = rfc_seal(e, _f_5);
      if (_mode_1 == 1) {
        Term _left_3 = term_ctr(CID_Con, _nd_4);
        u64 _nd_5 = heap_alloc(e, 1);
        e.mem[_nd_5 + 0] = rfc_seal(e, _f_6);
        e.mem[_nd_5 + 1] = rfc_seal(e, _reversed_1);
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
          u64 _t_10 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
          e.mem[_t_10 + 0] = 1ull;
          e.mem[_t_10 + 1] = _left_3;
          e.mem[_t_10 + 2] = _f_7;
          e.mem[_t_10 + 3] = term_ctr(CID_Con, _nd_5);
          e.mem[_t_10 + 4] = U32_BIN(_size_1, +, 1);
          return term_tsk(FID_ir_setMerge, _t_10);
        }
        r0 = 1ull;
        r1 = _left_3;
        r2 = _f_7;
        r3 = term_ctr(CID_Con, _nd_5);
        r4 = U32_BIN(_size_1, +, 1);
        WL_JMP(FID_ir_setMerge);
      } else {
        Term _left_4 = term_ctr(CID_Con, _nd_4);
        term_sink(e, _f_6);
        if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
          u64 _t_11 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
          e.mem[_t_11 + 0] = _mode_1;
          e.mem[_t_11 + 1] = _left_4;
          e.mem[_t_11 + 2] = _f_7;
          e.mem[_t_11 + 3] = _reversed_1;
          e.mem[_t_11 + 4] = _size_1;
          return term_tsk(FID_ir_setMerge, _t_11);
        }
        r0 = _mode_1;
        r1 = _left_4;
        r2 = _f_7;
        r3 = _reversed_1;
        r4 = _size_1;
        WL_JMP(FID_ir_setMerge);
      }
    }
  }}

  WL_CASE(FID_ir_valueLe)
    Term _a_0 = r0;
    Term _b_0 = r1;
    WL_OPEN
    if (seq) {
      WL_ROOM(1);
      STK(0) = FID_ir_valueLe_k1;
      WL_PUSHN(1);
    } else {
      u64 _t_0 = task_node(e, FID_ir_valueLe_k1, WL_CONT, WL_IDX, 1);
      WL_CONT = term_tsk(FID_ir_valueLe_k1, _t_0);
      WL_IDX = 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
      u64 _t_1 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _a_0;
      e.mem[_t_1 + 1] = _b_0;
      return term_tsk(FID_ir_valueOrder, _t_1);
    }
    r0 = _a_0;
    r1 = _b_0;
    WL_JMP(FID_ir_valueOrder);
  }}

  WL_CASE(FID_ir_valueLe_k1)
    u32 _h_0 = r0;
    WL_OPEN
    Term _o_0[1];
    if (spin_Cmp_is_le(e, _o_0, _h_0) == 0) {
      return 0;
    }
    r0 = (u32)_o_0[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_List_sort_pass_0)
    Term _n_0 = r0;
    Term _runs_0 = r1;
    WL_OPEN
    if (term_aux(_runs_0) == CID_Nil) {
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _runs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_1) == CID_Nil) {
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, term_pak(CID_Nil, 0));
        r0 = term_ctr(CID_Con, _nd_0);
        WL_RETN(1);
      } else {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _f_1, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        spare_free(e, cls_fit(2), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(3);
          STK(0) = _n_0;
          STK(1) = _f_3;
          STK(2) = FID_List_sort_pass_0_k1;
          WL_PUSHN(3);
        } else {
          u64 _t_0 = task_node(e, FID_List_sort_pass_0_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _n_0;
          e.mem[_t_0 + 1] = _f_3;
          WL_CONT = term_tsk(FID_List_sort_pass_0_k1, _t_0);
          WL_IDX = 2;
        }
        if (!DEVICE && !seq && fid_nofk(FID_List_merge_go_0)) {
          u64 _t_1 = task_node(e, FID_List_merge_go_0, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _n_0;
          e.mem[_t_1 + 1] = term_pak(CID_Nil, 0);
          e.mem[_t_1 + 2] = _f_0;
          e.mem[_t_1 + 3] = _f_2;
          return term_tsk(FID_List_merge_go_0, _t_1);
        }
        r0 = _n_0;
        r1 = term_pak(CID_Nil, 0);
        r2 = _f_0;
        r3 = _f_2;
        WL_JMP(FID_List_merge_go_0);
      }
    }
  }}

  WL_CASE(FID_List_sort_pass_0_k1)
    WL_POPN(2);
    Term _n_1 = STK(0);
    Term _f_4 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_0;
      STK(1) = FID_List_sort_pass_0_k3;
      WL_PUSHN(2);
    } else {
      u64 _t_2 = task_node(e, FID_List_sort_pass_0_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _h_0;
      WL_CONT = term_tsk(FID_List_sort_pass_0_k3, _t_2);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_List_sort_pass_0)) {
      u64 _t_3 = task_node(e, FID_List_sort_pass_0, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _n_1;
      e.mem[_t_3 + 1] = _f_4;
      return term_tsk(FID_List_sort_pass_0, _t_3);
    }
    r0 = _n_1;
    r1 = _f_4;
    WL_JMP(FID_List_sort_pass_0);
  }}

  WL_CASE(FID_List_sort_pass_0_k3)
    WL_POPN(1);
    Term _h_2 = STK(0);
    Term _h_1 = r0;
    WL_OPEN
    u64 _nd_1 = heap_alloc(e, 1);
    e.mem[_nd_1 + 0] = rfc_seal(e, _h_2);
    e.mem[_nd_1 + 1] = rfc_seal(e, _h_1);
    r0 = term_ctr(CID_Con, _nd_1);
    WL_RETN(1);
  }}

  WL_CASE(FID_List_concat)
    Term _xss_0 = r0;
    WL_OPEN
    WL_SPIN
    if (term_aux(_xss_0) == CID_Nil) {
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xss_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_0;
        STK(1) = FID_List_concat_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_List_concat_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        WL_CONT = term_tsk(FID_List_concat_k1, _t_0);
        WL_IDX = 1;
      }
      r0 = _f_1;
      _xss_0 = r0;
      WL_AGAIN(FID_List_concat);
    }
    WL_SPUN
  }}

  WL_CASE(FID_List_concat_k1)
    WL_POPN(1);
    Term _f_2 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_List_append)) {
      u64 _t_1 = task_node(e, FID_List_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _f_2;
      e.mem[_t_1 + 1] = _h_0;
      return term_tsk(FID_List_append, _t_1);
    }
    r0 = _f_2;
    r1 = _h_0;
    WL_JMP(FID_List_append);
  }}

  WL_CASE(FID_ir_domainSorted)
    Term _values_0 = r0;
    u32 _size_0 = r1;
    WL_OPEN
    if (term_aux(_values_0) == CID_Con) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VI) {
        u32 _first_0 = term_loc(_f_0);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(4);
          STK(0) = _first_0;
          STK(1) = _f_1;
          STK(2) = _size_0;
          STK(3) = FID_ir_domainSorted_k1;
          WL_PUSHN(4);
        } else {
          u64 _t_0 = task_node(e, FID_ir_domainSorted_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _first_0;
          e.mem[_t_0 + 1] = _f_1;
          e.mem[_t_0 + 2] = _size_0;
          WL_CONT = term_tsk(FID_ir_domainSorted_k1, _t_0);
          WL_IDX = 3;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_integerSequence)) {
          u64 _t_1 = task_node(e, FID_ir_integerSequence, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _first_0;
          e.mem[_t_1 + 1] = _f_1;
          return term_tsk(FID_ir_integerSequence, _t_1);
        }
        r0 = _first_0;
        r1 = _f_1;
        WL_JMP(FID_ir_integerSequence);
      } else {
        u64 _nd_2 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_0);
        e.mem[_nd_2 + 1] = rfc_seal(e, _f_1);
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_domainSorted_k5;
          WL_PUSHN(1);
        } else {
          u64 _t_4 = task_node(e, FID_ir_domainSorted_k5, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_domainSorted_k5, _t_4);
          WL_IDX = 0;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_buildDomainTree)) {
          u64 _t_5 = task_node(e, FID_ir_buildDomainTree, WL_CONT, WL_IDX, 0);
          e.mem[_t_5 + 0] = _size_0;
          e.mem[_t_5 + 1] = term_ctr(CID_Con, _nd_2);
          return term_tsk(FID_ir_buildDomainTree, _t_5);
        }
        r0 = _size_0;
        r1 = term_ctr(CID_Con, _nd_2);
        WL_JMP(FID_ir_buildDomainTree);
      }
    } else {
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_domainSorted_k7;
        WL_PUSHN(1);
      } else {
        u64 _t_6 = task_node(e, FID_ir_domainSorted_k7, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_domainSorted_k7, _t_6);
        WL_IDX = 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_buildDomainTree)) {
        u64 _t_7 = task_node(e, FID_ir_buildDomainTree, WL_CONT, WL_IDX, 0);
        e.mem[_t_7 + 0] = _size_0;
        e.mem[_t_7 + 1] = _values_0;
        return term_tsk(FID_ir_buildDomainTree, _t_7);
      }
      r0 = _size_0;
      r1 = _values_0;
      WL_JMP(FID_ir_buildDomainTree);
    }
  }}

  WL_CASE(FID_ir_domainSorted_k1)
    WL_POPN(3);
    u32 _first_1 = STK(0);
    Term _f_2 = STK(1);
    u32 _size_1 = STK(2);
    u32 _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, term_pak(CID_ir_VI, _first_1));
    e.mem[_nd_0 + 1] = rfc_seal(e, _f_2);
    if (_h_0 == 1) {
      Term _values_1 = term_ctr(CID_Con, _nd_0);
      term_sink(e, _values_1);
      u64 _nd_1 = heap_alloc(e, 1);
      e.mem[_nd_1 + 0] = rfc_seal(e, _first_1);
      e.mem[_nd_1 + 1] = rfc_seal(e, _size_1);
      r0 = term_ctr(CID_ir_IntDomain, _nd_1);
      WL_RETN(1);
    } else {
      Term _values_2 = term_ctr(CID_Con, _nd_0);
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_domainSorted_k3;
        WL_PUSHN(1);
      } else {
        u64 _t_2 = task_node(e, FID_ir_domainSorted_k3, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_domainSorted_k3, _t_2);
        WL_IDX = 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_buildDomainTree)) {
        u64 _t_3 = task_node(e, FID_ir_buildDomainTree, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _size_1;
        e.mem[_t_3 + 1] = _values_2;
        return term_tsk(FID_ir_buildDomainTree, _t_3);
      }
      r0 = _size_1;
      r1 = _values_2;
      WL_JMP(FID_ir_buildDomainTree);
    }
  }}

  WL_CASE(FID_ir_domainSorted_k3)
    Term _h_1 = r0;
    Term _h_2 = r1;
    WL_OPEN
    Term _o_0[1];
    if (spin_ir_domainBuilt(e, _o_0, _h_1, _h_2) == 0) {
      return 0;
    }
    r0 = _o_0[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_domainSorted_k5)
    Term _h_3 = r0;
    Term _h_4 = r1;
    WL_OPEN
    Term _o_1[1];
    if (spin_ir_domainBuilt(e, _o_1, _h_3, _h_4) == 0) {
      return 0;
    }
    r0 = _o_1[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_domainSorted_k7)
    Term _h_5 = r0;
    Term _h_6 = r1;
    WL_OPEN
    Term _o_2[1];
    if (spin_ir_domainBuilt(e, _o_2, _h_5, _h_6) == 0) {
      return 0;
    }
    r0 = _o_2[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionReplace)
    Term _tree_0 = r0;
    Term _key_0 = r1;
    Term _value_0 = r2;
    WL_OPEN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      term_sink(e, _key_0);
      term_sink(e, _value_0);
      r0 = term_pak(CID_ir_FunctionEmpty, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[6];
      u64 _sp_0 = ctr_take(e, _tree_0, 6, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _f_2 = _fb_0[2];
      Term _f_3 = _fb_0[3];
      u32 _f_4 = _fb_0[4];
      u32 _f_5 = _fb_0[5];
      _key_0 = term_keep(e, _key_0, 1);
      _f_1 = term_keep(e, _f_1, 1);
      spare_free(e, cls_fit(6), _sp_0);
      if (seq) {
        WL_ROOM(7);
        STK(0) = _f_0;
        STK(1) = _f_1;
        STK(2) = _f_2;
        STK(3) = _f_3;
        STK(4) = _key_0;
        STK(5) = _value_0;
        STK(6) = FID_ir_functionReplace_k1;
        WL_PUSHN(7);
      } else {
        u64 _t_0 = task_node(e, FID_ir_functionReplace_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        e.mem[_t_0 + 1] = _f_1;
        e.mem[_t_0 + 2] = _f_2;
        e.mem[_t_0 + 3] = _f_3;
        e.mem[_t_0 + 4] = _key_0;
        e.mem[_t_0 + 5] = _value_0;
        WL_CONT = term_tsk(FID_ir_functionReplace_k1, _t_0);
        WL_IDX = 6;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
        u64 _t_1 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _key_0;
        e.mem[_t_1 + 1] = _f_1;
        return term_tsk(FID_ir_valueOrder, _t_1);
      }
      r0 = _key_0;
      r1 = _f_1;
      WL_JMP(FID_ir_valueOrder);
    }
  }}

  WL_CASE(FID_ir_functionReplace_k1)
    WL_POPN(6);
    Term _f_6 = STK(0);
    Term _f_7 = STK(1);
    Term _f_8 = STK(2);
    Term _f_9 = STK(3);
    Term _key_1 = STK(4);
    Term _value_1 = STK(5);
    u32 _h_0 = r0;
    WL_OPEN
    if (_h_0 == 1) {
      term_sink(e, _f_8);
      term_sink(e, _key_1);
      Term _o_0[1];
      if (spin_ir_functionNode(e, _o_0, _f_6, _f_7, _value_1, _f_9) == 0) {
        return 0;
      }
      r0 = _o_0[0];
      WL_RETN(1);
    } else if (_h_0 == 0) {
      if (seq) {
        WL_ROOM(4);
        STK(0) = _f_7;
        STK(1) = _f_8;
        STK(2) = _f_9;
        STK(3) = FID_ir_functionReplace_k3;
        WL_PUSHN(4);
      } else {
        u64 _t_2 = task_node(e, FID_ir_functionReplace_k3, WL_CONT, WL_IDX, 1);
        e.mem[_t_2 + 0] = _f_7;
        e.mem[_t_2 + 1] = _f_8;
        e.mem[_t_2 + 2] = _f_9;
        WL_CONT = term_tsk(FID_ir_functionReplace_k3, _t_2);
        WL_IDX = 3;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionReplace)) {
        u64 _t_3 = task_node(e, FID_ir_functionReplace, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _f_6;
        e.mem[_t_3 + 1] = _key_1;
        e.mem[_t_3 + 2] = _value_1;
        return term_tsk(FID_ir_functionReplace, _t_3);
      }
      r0 = _f_6;
      r1 = _key_1;
      r2 = _value_1;
      WL_JMP(FID_ir_functionReplace);
    } else {
      if (seq) {
        WL_ROOM(4);
        STK(0) = _f_6;
        STK(1) = _f_7;
        STK(2) = _f_8;
        STK(3) = FID_ir_functionReplace_k5;
        WL_PUSHN(4);
      } else {
        u64 _t_4 = task_node(e, FID_ir_functionReplace_k5, WL_CONT, WL_IDX, 1);
        e.mem[_t_4 + 0] = _f_6;
        e.mem[_t_4 + 1] = _f_7;
        e.mem[_t_4 + 2] = _f_8;
        WL_CONT = term_tsk(FID_ir_functionReplace_k5, _t_4);
        WL_IDX = 3;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionReplace)) {
        u64 _t_5 = task_node(e, FID_ir_functionReplace, WL_CONT, WL_IDX, 0);
        e.mem[_t_5 + 0] = _f_9;
        e.mem[_t_5 + 1] = _key_1;
        e.mem[_t_5 + 2] = _value_1;
        return term_tsk(FID_ir_functionReplace, _t_5);
      }
      r0 = _f_9;
      r1 = _key_1;
      r2 = _value_1;
      WL_JMP(FID_ir_functionReplace);
    }
  }}

  WL_CASE(FID_ir_functionReplace_k3)
    WL_POPN(3);
    Term _f_10 = STK(0);
    Term _f_11 = STK(1);
    Term _f_12 = STK(2);
    Term _h_1 = r0;
    WL_OPEN
    Term _o_1[1];
    if (spin_ir_functionNode(e, _o_1, _h_1, _f_10, _f_11, _f_12) == 0) {
      return 0;
    }
    r0 = _o_1[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionReplace_k5)
    WL_POPN(3);
    Term _f_13 = STK(0);
    Term _f_14 = STK(1);
    Term _f_15 = STK(2);
    Term _h_2 = r0;
    WL_OPEN
    Term _o_2[1];
    if (spin_ir_functionNode(e, _o_2, _f_13, _f_14, _f_15, _h_2) == 0) {
      return 0;
    }
    r0 = _o_2[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_domainPartsOrder)
    u32 _order_0 = r0;
    Term _a_0 = r1;
    Term _b_0 = r2;
    WL_OPEN
    if (_order_0 == 1) {
      if (term_aux(_a_0) == CID_Nil) {
        if (term_aux(_b_0) == CID_Nil) {
          r0 = 1;
          WL_RETN(1);
        } else {
          term_sink(e, _b_0);
          r0 = 0;
          WL_RETN(1);
        }
      } else {
        Term _fb_0[2];
        u64 _sp_0 = ctr_take(e, _a_0, 2, _fb_0);
        Term _f_0 = _fb_0[0];
        Term _f_1 = _fb_0[1];
        if (term_aux(_b_0) == CID_Nil) {
          term_sink(e, _f_0);
          term_sink(e, _f_1);
          spare_free(e, cls_fit(2), _sp_0);
          r0 = 2;
          WL_RETN(1);
        } else {
          Term _fb_1[2];
          u64 _sp_1 = ctr_take(e, _b_0, 2, _fb_1);
          Term _f_2 = _fb_1[0];
          Term _f_3 = _fb_1[1];
          spare_free(e, cls_fit(2), _sp_1);
          spare_free(e, cls_fit(2), _sp_0);
          if (seq) {
            WL_ROOM(3);
            STK(0) = _f_1;
            STK(1) = _f_3;
            STK(2) = FID_ir_domainPartsOrder_k1;
            WL_PUSHN(3);
          } else {
            u64 _t_0 = task_node(e, FID_ir_domainPartsOrder_k1, WL_CONT, WL_IDX, 1);
            e.mem[_t_0 + 0] = _f_1;
            e.mem[_t_0 + 1] = _f_3;
            WL_CONT = term_tsk(FID_ir_domainPartsOrder_k1, _t_0);
            WL_IDX = 2;
          }
          if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
            u64 _t_1 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
            e.mem[_t_1 + 0] = _f_0;
            e.mem[_t_1 + 1] = _f_2;
            return term_tsk(FID_ir_valueOrder, _t_1);
          }
          r0 = _f_0;
          r1 = _f_2;
          WL_JMP(FID_ir_valueOrder);
        }
      }
    } else {
      term_sink(e, _a_0);
      term_sink(e, _b_0);
      r0 = _order_0;
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_domainPartsOrder_k1)
    WL_POPN(2);
    Term _f_4 = STK(0);
    Term _f_5 = STK(1);
    u32 _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_domainPartsOrder)) {
      u64 _t_2 = task_node(e, FID_ir_domainPartsOrder, WL_CONT, WL_IDX, 0);
      e.mem[_t_2 + 0] = _h_0;
      e.mem[_t_2 + 1] = _f_4;
      e.mem[_t_2 + 2] = _f_5;
      return term_tsk(FID_ir_domainPartsOrder, _t_2);
    }
    r0 = _h_0;
    r1 = _f_4;
    r2 = _f_5;
    WL_JMP(FID_ir_domainPartsOrder);
  }}

  WL_CASE(FID_ir_productCountFits)
    u32 _fits_0 = r0;
    Term _rest_0 = r1;
    u32 _count_0 = r2;
    u32 _size_0 = r3;
    WL_OPEN
    if (_fits_0 == 0) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_productCount)) {
        u64 _t_0 = task_node(e, FID_ir_productCount, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _rest_0;
        e.mem[_t_0 + 1] = U32_BIN(1048576ull, +, 1);
        return term_tsk(FID_ir_productCount, _t_0);
      }
      r0 = _rest_0;
      r1 = U32_BIN(1048576ull, +, 1);
      WL_JMP(FID_ir_productCount);
    } else {
      if (!DEVICE && !seq && fid_nofk(FID_ir_productCount)) {
        u64 _t_1 = task_node(e, FID_ir_productCount, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _rest_0;
        e.mem[_t_1 + 1] = U32_BIN(_count_0, *, _size_0);
        return term_tsk(FID_ir_productCount, _t_1);
      }
      r0 = _rest_0;
      r1 = U32_BIN(_count_0, *, _size_0);
      WL_JMP(FID_ir_productCount);
    }
  }}

  WL_CASE(FID_ir_productRows)
    Term _domains_0 = r0;
    WL_OPEN
    WL_SPIN
    if (term_aux(_domains_0) == CID_Nil) {
      r0 = term_ctr(CID_Con, STAT_OFF + 792);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _domains_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VSet) {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _f_0, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        term_sink(e, _f_3);
        spare_free(e, cls_fit(2), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(2);
          STK(0) = _f_2;
          STK(1) = FID_ir_productRows_k1;
          WL_PUSHN(2);
        } else {
          u64 _t_0 = task_node(e, FID_ir_productRows_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _f_2;
          WL_CONT = term_tsk(FID_ir_productRows_k1, _t_0);
          WL_IDX = 1;
        }
        r0 = _f_1;
        _domains_0 = r0;
        WL_AGAIN(FID_ir_productRows);
      } else {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = term_pak(CID_Nil, 0);
        WL_RETN(1);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_productRows_k1)
    WL_POPN(1);
    Term _f_4 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_productColumn)) {
      u64 _t_1 = task_node(e, FID_ir_productColumn, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _f_4;
      e.mem[_t_1 + 1] = _h_0;
      return term_tsk(FID_ir_productColumn, _t_1);
    }
    r0 = _f_4;
    r1 = _h_0;
    WL_JMP(FID_ir_productColumn);
  }}

  WL_CASE(FID_ir_productFunctions)
    Term _keys_0 = r0;
    Term _rows_0 = r1;
    WL_OPEN
    if (term_aux(_rows_0) == CID_Nil) {
      term_sink(e, _keys_0);
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _rows_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      _keys_0 = term_keep(e, _keys_0, 1);
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(3);
        STK(0) = _keys_0;
        STK(1) = _f_1;
        STK(2) = FID_ir_productFunctions_k1;
        WL_PUSHN(3);
      } else {
        u64 _t_0 = task_node(e, FID_ir_productFunctions_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _keys_0;
        e.mem[_t_0 + 1] = _f_1;
        WL_CONT = term_tsk(FID_ir_productFunctions_k1, _t_0);
        WL_IDX = 2;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionSorted)) {
        u64 _t_1 = task_node(e, FID_ir_functionSorted, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _keys_0;
        e.mem[_t_1 + 1] = _f_0;
        return term_tsk(FID_ir_functionSorted, _t_1);
      }
      r0 = _keys_0;
      r1 = _f_0;
      WL_JMP(FID_ir_functionSorted);
    }
  }}

  WL_CASE(FID_ir_productFunctions_k1)
    WL_POPN(2);
    Term _keys_1 = STK(0);
    Term _f_2 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_0;
      STK(1) = FID_ir_productFunctions_k3;
      WL_PUSHN(2);
    } else {
      u64 _t_2 = task_node(e, FID_ir_productFunctions_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _h_0;
      WL_CONT = term_tsk(FID_ir_productFunctions_k3, _t_2);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_productFunctions)) {
      u64 _t_3 = task_node(e, FID_ir_productFunctions, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _keys_1;
      e.mem[_t_3 + 1] = _f_2;
      return term_tsk(FID_ir_productFunctions, _t_3);
    }
    r0 = _keys_1;
    r1 = _f_2;
    WL_JMP(FID_ir_productFunctions);
  }}

  WL_CASE(FID_ir_productFunctions_k3)
    WL_POPN(1);
    Term _h_2 = STK(0);
    Term _h_1 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _h_2);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_1);
    r0 = term_ctr(CID_Con, _nd_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_filteredFinish)
    Term _reversed_0 = r0;
    Term _sorted_0 = r1;
    u32 _size_0 = r2;
    WL_OPEN
    WL_SPIN
    if (term_aux(_reversed_0) == CID_Nil) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
        u64 _t_0 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _sorted_0;
        e.mem[_t_0 + 1] = _size_0;
        return term_tsk(FID_ir_setSorted, _t_0);
      }
      r0 = _sorted_0;
      r1 = _size_0;
      WL_JMP(FID_ir_setSorted);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _reversed_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, _sorted_0);
      r0 = _f_1;
      r1 = term_ctr(CID_Con, _nd_0);
      r2 = U32_BIN(_size_0, +, 1);
      _reversed_0 = r0;
      _sorted_0 = r1;
      _size_0 = r2;
      WL_AGAIN(FID_ir_filteredFinish);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_powerCountFits)
    u32 _fits_0 = r0;
    u32 _remaining_0 = r1;
    u32 _size_0 = r2;
    WL_OPEN
    if (_fits_0 == 0) {
      r0 = 0;
      r1 = 0;
      WL_RETN(2);
    } else {
      if (!DEVICE && !seq && fid_nofk(FID_ir_powerCount)) {
        u64 _t_0 = task_node(e, FID_ir_powerCount, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _remaining_0;
        e.mem[_t_0 + 1] = _size_0;
        return term_tsk(FID_ir_powerCount, _t_0);
      }
      r0 = _remaining_0;
      r1 = _size_0;
      WL_JMP(FID_ir_powerCount);
    }
  }}

  WL_CASE(FID_ir_powerItems)
    Term _values_0 = r0;
    WL_OPEN
    WL_SPIN
    if (term_aux(_values_0) == CID_Nil) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
        u64 _t_0 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = term_ctr(CID_Con, STAT_OFF + 797);
        e.mem[_t_0 + 1] = 1ull;
        return term_tsk(FID_ir_setSorted, _t_0);
      }
      r0 = term_ctr(CID_Con, STAT_OFF + 797);
      r1 = 1ull;
      WL_JMP(FID_ir_setSorted);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_0;
        STK(1) = FID_ir_powerItems_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_1 = task_node(e, FID_ir_powerItems_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_1 + 0] = _f_0;
        WL_CONT = term_tsk(FID_ir_powerItems_k1, _t_1);
        WL_IDX = 1;
      }
      r0 = _f_1;
      _values_0 = r0;
      WL_AGAIN(FID_ir_powerItems);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_powerItems_k1)
    WL_POPN(1);
    Term _f_2 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    if (term_aux(_h_0) == CID_ir_VE) {
      term_sink(e, _f_2);
      Term _fb_1[1];
      u64 _sp_1 = ctr_take(e, _h_0, 1, _fb_1);
      Term _f_3 = _fb_1[0];
      u64 _nd_0 = _sp_1 >= HEAP_OFF ? _sp_1 : heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_3);
      r0 = term_ctr(CID_ir_VE, _nd_0);
      WL_RETN(1);
    } else if (term_aux(_h_0) == CID_ir_VSet) {
      Term _fb_2[2];
      u64 _sp_2 = ctr_take(e, _h_0, 2, _fb_2);
      Term _f_4 = _fb_2[0];
      Term _f_5 = _fb_2[1];
      term_sink(e, _f_5);
      _f_4 = term_keep(e, _f_4, 1);
      spare_free(e, cls_fit(2), _sp_2);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_4;
        STK(1) = FID_ir_powerItems_k3;
        WL_PUSHN(2);
      } else {
        u64 _t_2 = task_node(e, FID_ir_powerItems_k3, WL_CONT, WL_IDX, 1);
        e.mem[_t_2 + 0] = _f_4;
        WL_CONT = term_tsk(FID_ir_powerItems_k3, _t_2);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_powerAdd)) {
        u64 _t_3 = task_node(e, FID_ir_powerAdd, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _f_2;
        e.mem[_t_3 + 1] = _f_4;
        return term_tsk(FID_ir_powerAdd, _t_3);
      }
      r0 = _f_2;
      r1 = _f_4;
      WL_JMP(FID_ir_powerAdd);
    } else {
      term_sink(e, _f_2);
      term_sink(e, _h_0);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 519);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_powerItems_k3)
    WL_POPN(1);
    Term _f_6 = STK(0);
    Term _h_1 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
      u64 _t_4 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
      e.mem[_t_4 + 0] = 1ull;
      e.mem[_t_4 + 1] = _f_6;
      e.mem[_t_4 + 2] = _h_1;
      e.mem[_t_4 + 3] = term_pak(CID_Nil, 0);
      e.mem[_t_4 + 4] = 0ull;
      return term_tsk(FID_ir_setMerge, _t_4);
    }
    r0 = 1ull;
    r1 = _f_6;
    r2 = _h_1;
    r3 = term_pak(CID_Nil, 0);
    r4 = 0ull;
    WL_JMP(FID_ir_setMerge);
  }}

  WL_CASE(FID_ir_setBinaryLists)
    u32 _op_0 = r0;
    Term _left_0 = r1;
    Term _right_0 = r2;
    WL_OPEN
    if (_op_0 == 1) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
        u64 _t_0 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = 1ull;
        e.mem[_t_0 + 1] = _left_0;
        e.mem[_t_0 + 2] = _right_0;
        e.mem[_t_0 + 3] = term_pak(CID_Nil, 0);
        e.mem[_t_0 + 4] = 0ull;
        return term_tsk(FID_ir_setMerge, _t_0);
      }
      r0 = 1ull;
      r1 = _left_0;
      r2 = _right_0;
      r3 = term_pak(CID_Nil, 0);
      r4 = 0ull;
      WL_JMP(FID_ir_setMerge);
    } else if ((_op_0 & 3) == 1) {
      term_sink(e, _left_0);
      term_sink(e, _right_0);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 878);
      WL_RETN(1);
    } else if (_op_0 == 3) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
        u64 _t_1 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = 3ull;
        e.mem[_t_1 + 1] = _left_0;
        e.mem[_t_1 + 2] = _right_0;
        e.mem[_t_1 + 3] = term_pak(CID_Nil, 0);
        e.mem[_t_1 + 4] = 0ull;
        return term_tsk(FID_ir_setMerge, _t_1);
      }
      r0 = 3ull;
      r1 = _left_0;
      r2 = _right_0;
      r3 = term_pak(CID_Nil, 0);
      r4 = 0ull;
      WL_JMP(FID_ir_setMerge);
    } else if ((_op_0 & 3) == 3) {
      term_sink(e, _left_0);
      term_sink(e, _right_0);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 878);
      WL_RETN(1);
    } else if (_op_0 == 2) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setMerge)) {
        u64 _t_2 = task_node(e, FID_ir_setMerge, WL_CONT, WL_IDX, 0);
        e.mem[_t_2 + 0] = 2ull;
        e.mem[_t_2 + 1] = _left_0;
        e.mem[_t_2 + 2] = _right_0;
        e.mem[_t_2 + 3] = term_pak(CID_Nil, 0);
        e.mem[_t_2 + 4] = 0ull;
        return term_tsk(FID_ir_setMerge, _t_2);
      }
      r0 = 2ull;
      r1 = _left_0;
      r2 = _right_0;
      r3 = term_pak(CID_Nil, 0);
      r4 = 0ull;
      WL_JMP(FID_ir_setMerge);
    } else if ((_op_0 & 3) == 2) {
      term_sink(e, _left_0);
      term_sink(e, _right_0);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 878);
      WL_RETN(1);
    } else if (_op_0 == 4) {
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_setBinaryLists_k1;
        WL_PUSHN(1);
      } else {
        u64 _t_3 = task_node(e, FID_ir_setBinaryLists_k1, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_setBinaryLists_k1, _t_3);
        WL_IDX = 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSubset)) {
        u64 _t_4 = task_node(e, FID_ir_setSubset, WL_CONT, WL_IDX, 0);
        e.mem[_t_4 + 0] = _left_0;
        e.mem[_t_4 + 1] = _right_0;
        return term_tsk(FID_ir_setSubset, _t_4);
      }
      r0 = _left_0;
      r1 = _right_0;
      WL_JMP(FID_ir_setSubset);
    } else {
      term_sink(e, _left_0);
      term_sink(e, _right_0);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 878);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_setBinaryLists_k1)
    u32 _h_0 = r0;
    WL_OPEN
    r0 = term_pak(CID_ir_VB, _h_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_domainTreeIndex)
    Term _tree_0 = r0;
    Term _value_0 = r1;
    u32 _offset_0 = r2;
    WL_OPEN
    if (term_aux(_tree_0) == CID_ir_DomainEmpty) {
      term_sink(e, _value_0);
      r0 = 0;
      r1 = 0;
      WL_RETN(2);
    } else {
      Term _fb_0[4];
      u64 _sp_0 = ctr_take(e, _tree_0, 4, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _f_2 = _fb_0[2];
      u32 _f_3 = _fb_0[3];
      _value_0 = term_keep(e, _value_0, 1);
      spare_free(e, cls_fit(4), _sp_0);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _f_0;
        STK(1) = _f_2;
        STK(2) = _value_0;
        STK(3) = _offset_0;
        STK(4) = FID_ir_domainTreeIndex_k1;
        WL_PUSHN(5);
      } else {
        u64 _t_0 = task_node(e, FID_ir_domainTreeIndex_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        e.mem[_t_0 + 1] = _f_2;
        e.mem[_t_0 + 2] = _value_0;
        e.mem[_t_0 + 3] = _offset_0;
        WL_CONT = term_tsk(FID_ir_domainTreeIndex_k1, _t_0);
        WL_IDX = 4;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
        u64 _t_1 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _value_0;
        e.mem[_t_1 + 1] = _f_1;
        return term_tsk(FID_ir_valueOrder, _t_1);
      }
      r0 = _value_0;
      r1 = _f_1;
      WL_JMP(FID_ir_valueOrder);
    }
  }}

  WL_CASE(FID_ir_domainTreeIndex_k1)
    WL_POPN(4);
    Term _f_4 = STK(0);
    Term _f_5 = STK(1);
    Term _value_1 = STK(2);
    u32 _offset_1 = STK(3);
    u32 _h_0 = r0;
    WL_OPEN
    Term _o_0[1];
    if (spin_ir_domainTreeSize(e, _o_0, _f_4) == 0) {
      return 0;
    }
    if (_h_0 == 0) {
      term_sink(e, _f_5);
      u32 _middle_0 = U32_BIN(_offset_1, +, (u32)_o_0[0]);
      if (!DEVICE && !seq && fid_nofk(FID_ir_domainTreeIndex)) {
        u64 _t_2 = task_node(e, FID_ir_domainTreeIndex, WL_CONT, WL_IDX, 0);
        e.mem[_t_2 + 0] = _f_4;
        e.mem[_t_2 + 1] = _value_1;
        e.mem[_t_2 + 2] = _offset_1;
        return term_tsk(FID_ir_domainTreeIndex, _t_2);
      }
      r0 = _f_4;
      r1 = _value_1;
      r2 = _offset_1;
      WL_JMP(FID_ir_domainTreeIndex);
    } else if (_h_0 == 1) {
      term_sink(e, _f_4);
      term_sink(e, _f_5);
      term_sink(e, _value_1);
      u32 _middle_1 = U32_BIN(_offset_1, +, (u32)_o_0[0]);
      r0 = 1;
      r1 = _middle_1;
      WL_RETN(2);
    } else {
      term_sink(e, _f_4);
      u32 _middle_2 = U32_BIN(_offset_1, +, (u32)_o_0[0]);
      if (!DEVICE && !seq && fid_nofk(FID_ir_domainTreeIndex)) {
        u64 _t_3 = task_node(e, FID_ir_domainTreeIndex, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _f_5;
        e.mem[_t_3 + 1] = _value_1;
        e.mem[_t_3 + 2] = U32_BIN(_middle_2, +, 1);
        return term_tsk(FID_ir_domainTreeIndex, _t_3);
      }
      r0 = _f_5;
      r1 = _value_1;
      r2 = U32_BIN(_middle_2, +, 1);
      WL_JMP(FID_ir_domainTreeIndex);
    }
  }}

  WL_CASE(FID_List_sort_pairs_0)
    Term _xs_0 = r0;
    WL_OPEN
    if (term_aux(_xs_0) == CID_Nil) {
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_1) == CID_Nil) {
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, term_pak(CID_Nil, 0));
        u64 _nd_1 = heap_alloc(e, 1);
        e.mem[_nd_1 + 0] = rfc_seal(e, term_ctr(CID_Con, _nd_0));
        e.mem[_nd_1 + 1] = rfc_seal(e, term_pak(CID_Nil, 0));
        r0 = term_ctr(CID_Con, _nd_1);
        WL_RETN(1);
      } else {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _f_1, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        _f_0 = term_keep(e, _f_0, 1);
        _f_2 = term_keep(e, _f_2, 1);
        spare_free(e, cls_fit(2), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(4);
          STK(0) = _f_0;
          STK(1) = _f_2;
          STK(2) = _f_3;
          STK(3) = FID_List_sort_pairs_0_k1;
          WL_PUSHN(4);
        } else {
          u64 _t_0 = task_node(e, FID_List_sort_pairs_0_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _f_0;
          e.mem[_t_0 + 1] = _f_2;
          e.mem[_t_0 + 2] = _f_3;
          WL_CONT = term_tsk(FID_List_sort_pairs_0_k1, _t_0);
          WL_IDX = 3;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_valueLe)) {
          u64 _t_1 = task_node(e, FID_ir_valueLe, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _f_0;
          e.mem[_t_1 + 1] = _f_2;
          return term_tsk(FID_ir_valueLe, _t_1);
        }
        r0 = _f_0;
        r1 = _f_2;
        WL_JMP(FID_ir_valueLe);
      }
    }
  }}

  WL_CASE(FID_List_sort_pairs_0_k1)
    WL_POPN(3);
    Term _f_4 = STK(0);
    Term _f_5 = STK(1);
    Term _f_6 = STK(2);
    u32 _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _f_4;
      STK(1) = _f_5;
      STK(2) = _h_0;
      STK(3) = FID_List_sort_pairs_0_k3;
      WL_PUSHN(4);
    } else {
      u64 _t_2 = task_node(e, FID_List_sort_pairs_0_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _f_4;
      e.mem[_t_2 + 1] = _f_5;
      e.mem[_t_2 + 2] = _h_0;
      WL_CONT = term_tsk(FID_List_sort_pairs_0_k3, _t_2);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_List_sort_pairs_0)) {
      u64 _t_3 = task_node(e, FID_List_sort_pairs_0, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _f_6;
      return term_tsk(FID_List_sort_pairs_0, _t_3);
    }
    r0 = _f_6;
    WL_JMP(FID_List_sort_pairs_0);
  }}

  WL_CASE(FID_List_sort_pairs_0_k3)
    WL_POPN(3);
    Term _f_7 = STK(0);
    Term _f_8 = STK(1);
    u32 _h_2 = STK(2);
    Term _h_1 = r0;
    WL_OPEN
    Term _o_0[1];
    if (spin_List_sort_pair(e, _o_0, _f_7, _f_8, _h_2) == 0) {
      return 0;
    }
    u64 _nd_2 = heap_alloc(e, 1);
    e.mem[_nd_2 + 0] = rfc_seal(e, _o_0[0]);
    e.mem[_nd_2 + 1] = rfc_seal(e, _h_1);
    r0 = term_ctr(CID_Con, _nd_2);
    WL_RETN(1);
  }}

  WL_CASE(FID_List_sort_go_0)
    Term _fuel_0 = r0;
    Term _n_0 = r1;
    Term _runs_0 = r2;
    WL_OPEN
    if (_fuel_0 == 0) {
      if (!DEVICE && !seq && fid_nofk(FID_List_concat)) {
        u64 _t_0 = task_node(e, FID_List_concat, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _runs_0;
        return term_tsk(FID_List_concat, _t_0);
      }
      r0 = _runs_0;
      WL_JMP(FID_List_concat);
    } else {
      Term _f_0 = (_fuel_0 - 1);
      if (term_aux(_runs_0) == CID_Nil) {
        r0 = term_pak(CID_Nil, 0);
        WL_RETN(1);
      } else {
        Term _fb_0[2];
        u64 _sp_0 = ctr_take(e, _runs_0, 2, _fb_0);
        Term _f_1 = _fb_0[0];
        Term _f_2 = _fb_0[1];
        if (term_aux(_f_2) == CID_Nil) {
          spare_free(e, cls_fit(2), _sp_0);
          r0 = _f_1;
          WL_RETN(1);
        } else {
          Term _fb_1[2];
          u64 _sp_1 = ctr_take(e, _f_2, 2, _fb_1);
          Term _f_3 = _fb_1[0];
          Term _f_4 = _fb_1[1];
          u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
          e.mem[_nd_0 + 0] = rfc_seal(e, _f_3);
          e.mem[_nd_0 + 1] = rfc_seal(e, _f_4);
          u64 _nd_1 = _sp_1 >= HEAP_OFF ? _sp_1 : heap_alloc(e, 1);
          e.mem[_nd_1 + 0] = rfc_seal(e, _f_1);
          e.mem[_nd_1 + 1] = rfc_seal(e, term_ctr(CID_Con, _nd_0));
          if (seq) {
            WL_ROOM(3);
            STK(0) = _f_0;
            STK(1) = _n_0;
            STK(2) = FID_List_sort_go_0_k1;
            WL_PUSHN(3);
          } else {
            u64 _t_1 = task_node(e, FID_List_sort_go_0_k1, WL_CONT, WL_IDX, 1);
            e.mem[_t_1 + 0] = _f_0;
            e.mem[_t_1 + 1] = _n_0;
            WL_CONT = term_tsk(FID_List_sort_go_0_k1, _t_1);
            WL_IDX = 2;
          }
          if (!DEVICE && !seq && fid_nofk(FID_List_sort_pass_0)) {
            u64 _t_2 = task_node(e, FID_List_sort_pass_0, WL_CONT, WL_IDX, 0);
            e.mem[_t_2 + 0] = _n_0;
            e.mem[_t_2 + 1] = term_ctr(CID_Con, _nd_1);
            return term_tsk(FID_List_sort_pass_0, _t_2);
          }
          r0 = _n_0;
          r1 = term_ctr(CID_Con, _nd_1);
          WL_JMP(FID_List_sort_pass_0);
        }
      }
    }
  }}

  WL_CASE(FID_List_sort_go_0_k1)
    WL_POPN(2);
    Term _f_5 = STK(0);
    Term _n_1 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_List_sort_go_0)) {
      u64 _t_3 = task_node(e, FID_List_sort_go_0, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _f_5;
      e.mem[_t_3 + 1] = _n_1;
      e.mem[_t_3 + 2] = _h_0;
      return term_tsk(FID_List_sort_go_0, _t_3);
    }
    r0 = _f_5;
    r1 = _n_1;
    r2 = _h_0;
    WL_JMP(FID_List_sort_go_0);
  }}

  WL_CASE(FID_List_length)
    Term _xs_0 = r0;
    WL_OPEN
    WL_SPIN
    if (term_aux(_xs_0) == CID_Nil) {
      r0 = 0ull;
      WL_RETN(1);
    } else {
      u64 _sp_0 = term_peek(e.mem, _xs_0);
      Term _f_0 = e.mem[_sp_0 + 0];
      Term _f_1 = e.mem[_sp_0 + 1];
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_List_length_k1;
        WL_PUSHN(1);
      } else {
        u64 _t_0 = task_node(e, FID_List_length_k1, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_List_length_k1, _t_0);
        WL_IDX = 0;
      }
      r0 = _f_1;
      _xs_0 = r0;
      WL_AGAIN(FID_List_length);
    }
    WL_SPUN
  }}

  WL_CASE(FID_List_length_k1)
    Term _h_0 = r0;
    WL_OPEN
    r0 = nat_chk(e, _h_0 + 1);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionOrder)
    Term _left_0 = r0;
    Term _right_0 = r1;
    WL_OPEN
    Term _o_0[1];
    if (spin_ir_functionSize(e, _o_0, _left_0) == 0) {
      return 0;
    }
    Term _o_1[1];
    if (spin_ir_functionSize(e, _o_1, _right_0) == 0) {
      return 0;
    }
    u32 _s_0 = (U32_BIN((u32)_o_0[0], >, (u32)_o_1[0]) + U32_BIN((u32)_o_0[0], >=, (u32)_o_1[0]));
    if (_s_0 == 1) {
      Term _o_2[1];
      if (spin_ir_functionStack(e, _o_2, _left_0, term_pak(CID_Nil, 0)) == 0) {
        return 0;
      }
      Term _o_3[1];
      if (spin_ir_functionStack(e, _o_3, _right_0, term_pak(CID_Nil, 0)) == 0) {
        return 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionOrderStacks)) {
        u64 _t_0 = task_node(e, FID_ir_functionOrderStacks, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _o_2[0];
        e.mem[_t_0 + 1] = _o_3[0];
        return term_tsk(FID_ir_functionOrderStacks, _t_0);
      }
      r0 = _o_2[0];
      r1 = _o_3[0];
      WL_JMP(FID_ir_functionOrderStacks);
    } else {
      term_sink(e, _left_0);
      term_sink(e, _right_0);
      r0 = _s_0;
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_listEq)
    Term _xs_0 = r0;
    Term _ys_0 = r1;
    WL_OPEN
    if (term_aux(_xs_0) == CID_Nil) {
      if (term_aux(_ys_0) == CID_Nil) {
        r0 = 1;
        WL_RETN(1);
      } else {
        term_sink(e, _ys_0);
        r0 = 0;
        WL_RETN(1);
      }
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_ys_0) == CID_Nil) {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = 0;
        WL_RETN(1);
      } else {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _ys_0, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        spare_free(e, cls_fit(2), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(3);
          STK(0) = _f_1;
          STK(1) = _f_3;
          STK(2) = FID_ir_listEq_k1;
          WL_PUSHN(3);
        } else {
          u64 _t_0 = task_node(e, FID_ir_listEq_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _f_1;
          e.mem[_t_0 + 1] = _f_3;
          WL_CONT = term_tsk(FID_ir_listEq_k1, _t_0);
          WL_IDX = 2;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_veq)) {
          u64 _t_1 = task_node(e, FID_ir_veq, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _f_0;
          e.mem[_t_1 + 1] = _f_2;
          return term_tsk(FID_ir_veq, _t_1);
        }
        r0 = _f_0;
        r1 = _f_2;
        WL_JMP(FID_ir_veq);
      }
    }
  }}

  WL_CASE(FID_ir_listEq_k1)
    WL_POPN(2);
    Term _f_4 = STK(0);
    Term _f_5 = STK(1);
    u32 _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_0;
      STK(1) = FID_ir_listEq_k3;
      WL_PUSHN(2);
    } else {
      u64 _t_2 = task_node(e, FID_ir_listEq_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _h_0;
      WL_CONT = term_tsk(FID_ir_listEq_k3, _t_2);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_listEq)) {
      u64 _t_3 = task_node(e, FID_ir_listEq, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _f_4;
      e.mem[_t_3 + 1] = _f_5;
      return term_tsk(FID_ir_listEq, _t_3);
    }
    r0 = _f_4;
    r1 = _f_5;
    WL_JMP(FID_ir_listEq);
  }}

  WL_CASE(FID_ir_listEq_k3)
    WL_POPN(1);
    u32 _h_2 = STK(0);
    u32 _h_1 = r0;
    WL_OPEN
    Term _o_0[1];
    if (spin_Bool_and(e, _o_0, _h_2, _h_1) == 0) {
      return 0;
    }
    r0 = (u32)_o_0[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_valueOrder)
    Term _a_0 = r0;
    Term _b_0 = r1;
    WL_OPEN
    if (term_aux(_a_0) == CID_ir_VI) {
      u32 _x_0 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VI) {
        u32 _y_0 = term_loc(_b_0);
        Term _o_0[1];
        if (spin_ir_signedOrder(e, _o_0, _x_0, _y_0) == 0) {
          return 0;
        }
        r0 = (u32)_o_0[0];
        WL_RETN(1);
      } else {
        Term _o_1[1];
        if (spin_ir_valueTag(e, _o_1, term_pak(CID_ir_VI, _x_0)) == 0) {
          return 0;
        }
        Term _o_2[1];
        if (spin_ir_valueTag(e, _o_2, _b_0) == 0) {
          return 0;
        }
        r0 = (U32_BIN((u32)_o_1[0], >, (u32)_o_2[0]) + U32_BIN((u32)_o_1[0], >=, (u32)_o_2[0]));
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VB) {
      u32 _x_1 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VB) {
        u32 _y_1 = term_loc(_b_0);
        Term _o_3[1];
        if (spin_Bool_to_u32(e, _o_3, _x_1) == 0) {
          return 0;
        }
        Term _o_4[1];
        if (spin_Bool_to_u32(e, _o_4, _y_1) == 0) {
          return 0;
        }
        r0 = (U32_BIN((u32)_o_3[0], >, (u32)_o_4[0]) + U32_BIN((u32)_o_3[0], >=, (u32)_o_4[0]));
        WL_RETN(1);
      } else {
        Term _o_5[1];
        if (spin_ir_valueTag(e, _o_5, term_pak(CID_ir_VB, _x_1)) == 0) {
          return 0;
        }
        Term _o_6[1];
        if (spin_ir_valueTag(e, _o_6, _b_0) == 0) {
          return 0;
        }
        r0 = (U32_BIN((u32)_o_5[0], >, (u32)_o_6[0]) + U32_BIN((u32)_o_5[0], >=, (u32)_o_6[0]));
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VS) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_b_0) == CID_ir_VS) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _b_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        Term _o_7[1];
        if (spin_String_order(e, _o_7, _f_0, _f_1) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(1), _sp_1);
        spare_free(e, cls_fit(1), _sp_0);
        r0 = (u32)_o_7[0];
        WL_RETN(1);
      } else {
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        Term _o_8[1];
        if (spin_ir_valueTag(e, _o_8, term_ctr(CID_ir_VS, _nd_0)) == 0) {
          return 0;
        }
        Term _o_9[1];
        if (spin_ir_valueTag(e, _o_9, _b_0) == 0) {
          return 0;
        }
        r0 = (U32_BIN((u32)_o_8[0], >, (u32)_o_9[0]) + U32_BIN((u32)_o_8[0], >=, (u32)_o_9[0]));
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VSet) {
      Term _fb_2[2];
      u64 _sp_2 = ctr_take(e, _a_0, 2, _fb_2);
      Term _f_2 = _fb_2[0];
      Term _f_3 = _fb_2[1];
      if (term_aux(_b_0) == CID_ir_VSet) {
        Term _fb_3[2];
        u64 _sp_3 = ctr_take(e, _b_0, 2, _fb_3);
        Term _f_4 = _fb_3[0];
        Term _f_5 = _fb_3[1];
        Term _o_10[1];
        if (spin_ir_domainSize(e, _o_10, _f_3) == 0) {
          return 0;
        }
        Term _o_11[1];
        if (spin_ir_domainSize(e, _o_11, _f_5) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(2), _sp_3);
        spare_free(e, cls_fit(2), _sp_2);
        if (!DEVICE && !seq && fid_nofk(FID_ir_domainPartsOrder)) {
          u64 _t_0 = task_node(e, FID_ir_domainPartsOrder, WL_CONT, WL_IDX, 0);
          e.mem[_t_0 + 0] = (U32_BIN((u32)_o_10[0], >, (u32)_o_11[0]) + U32_BIN((u32)_o_10[0], >=, (u32)_o_11[0]));
          e.mem[_t_0 + 1] = _f_2;
          e.mem[_t_0 + 2] = _f_4;
          return term_tsk(FID_ir_domainPartsOrder, _t_0);
        }
        r0 = (U32_BIN((u32)_o_10[0], >, (u32)_o_11[0]) + U32_BIN((u32)_o_10[0], >=, (u32)_o_11[0]));
        r1 = _f_2;
        r2 = _f_4;
        WL_JMP(FID_ir_domainPartsOrder);
      } else {
        u64 _nd_1 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_2);
        e.mem[_nd_1 + 1] = rfc_seal(e, _f_3);
        Term _o_12[1];
        if (spin_ir_valueTag(e, _o_12, term_ctr(CID_ir_VSet, _nd_1)) == 0) {
          return 0;
        }
        Term _o_13[1];
        if (spin_ir_valueTag(e, _o_13, _b_0) == 0) {
          return 0;
        }
        r0 = (U32_BIN((u32)_o_12[0], >, (u32)_o_13[0]) + U32_BIN((u32)_o_12[0], >=, (u32)_o_13[0]));
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VFun) {
      Term _fb_4[2];
      u64 _sp_4 = ctr_take(e, _a_0, 2, _fb_4);
      Term _f_6 = _fb_4[0];
      u32 _f_7 = _fb_4[1];
      if (term_aux(_b_0) == CID_ir_VFun) {
        Term _fb_5[2];
        u64 _sp_5 = ctr_take(e, _b_0, 2, _fb_5);
        Term _f_8 = _fb_5[0];
        u32 _f_9 = _fb_5[1];
        spare_free(e, cls_fit(2), _sp_5);
        spare_free(e, cls_fit(2), _sp_4);
        if (!DEVICE && !seq && fid_nofk(FID_ir_functionOrder)) {
          u64 _t_1 = task_node(e, FID_ir_functionOrder, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _f_6;
          e.mem[_t_1 + 1] = _f_8;
          return term_tsk(FID_ir_functionOrder, _t_1);
        }
        r0 = _f_6;
        r1 = _f_8;
        WL_JMP(FID_ir_functionOrder);
      } else {
        u64 _nd_2 = _sp_4 >= HEAP_OFF ? _sp_4 : heap_alloc(e, 1);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_6);
        e.mem[_nd_2 + 1] = rfc_seal(e, _f_7);
        Term _o_14[1];
        if (spin_ir_valueTag(e, _o_14, term_ctr(CID_ir_VFun, _nd_2)) == 0) {
          return 0;
        }
        Term _o_15[1];
        if (spin_ir_valueTag(e, _o_15, _b_0) == 0) {
          return 0;
        }
        r0 = (U32_BIN((u32)_o_14[0], >, (u32)_o_15[0]) + U32_BIN((u32)_o_14[0], >=, (u32)_o_15[0]));
        WL_RETN(1);
      }
    } else {
      Term _fb_6[1];
      u64 _sp_6 = ctr_take(e, _a_0, 1, _fb_6);
      Term _f_10 = _fb_6[0];
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_7[1];
        u64 _sp_7 = ctr_take(e, _b_0, 1, _fb_7);
        Term _f_11 = _fb_7[0];
        Term _o_16[1];
        if (spin_String_order(e, _o_16, _f_10, _f_11) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(1), _sp_7);
        spare_free(e, cls_fit(1), _sp_6);
        r0 = (u32)_o_16[0];
        WL_RETN(1);
      } else {
        u64 _nd_3 = _sp_6 >= HEAP_OFF ? _sp_6 : heap_alloc(e, 0);
        e.mem[_nd_3 + 0] = rfc_seal(e, _f_10);
        Term _o_17[1];
        if (spin_ir_valueTag(e, _o_17, term_ctr(CID_ir_VE, _nd_3)) == 0) {
          return 0;
        }
        Term _o_18[1];
        if (spin_ir_valueTag(e, _o_18, _b_0) == 0) {
          return 0;
        }
        r0 = (U32_BIN((u32)_o_17[0], >, (u32)_o_18[0]) + U32_BIN((u32)_o_17[0], >=, (u32)_o_18[0]));
        WL_RETN(1);
      }
    }
  }}

  WL_CASE(FID_ir_vlen)
    Term _xs_0 = r0;
    WL_OPEN
    WL_SPIN
    if (term_aux(_xs_0) == CID_Nil) {
      r0 = 0ull;
      WL_RETN(1);
    } else {
      u64 _sp_0 = term_peek(e.mem, _xs_0);
      Term _f_0 = e.mem[_sp_0 + 0];
      Term _f_1 = e.mem[_sp_0 + 1];
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_vlen_k1;
        WL_PUSHN(1);
      } else {
        u64 _t_0 = task_node(e, FID_ir_vlen_k1, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_vlen_k1, _t_0);
        WL_IDX = 0;
      }
      r0 = _f_1;
      _xs_0 = r0;
      WL_AGAIN(FID_ir_vlen);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_vlen_k1)
    u32 _h_0 = r0;
    WL_OPEN
    r0 = U32_BIN(_h_0, +, 1);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_powerCount)
    u32 _remaining_0 = r0;
    u32 _size_0 = r1;
    WL_OPEN
    if (_remaining_0 == 0) {
      r0 = 1;
      r1 = _size_0;
      WL_RETN(2);
    } else {
      Term _a_0 = ((u32)(2ull) == 0 ? 0 : (u64)U32_QUO((u32)(1048576ull), (u32)(2ull)));
      if (!DEVICE && !seq && fid_nofk(FID_ir_powerCountFits)) {
        u64 _t_0 = task_node(e, FID_ir_powerCountFits, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = U32_BIN(_size_0, <=, _a_0);
        e.mem[_t_0 + 1] = U32_BIN(_remaining_0, -, 1ull);
        e.mem[_t_0 + 2] = U32_BIN(_size_0, +, _size_0);
        return term_tsk(FID_ir_powerCountFits, _t_0);
      }
      r0 = U32_BIN(_size_0, <=, _a_0);
      r1 = U32_BIN(_remaining_0, -, 1ull);
      r2 = U32_BIN(_size_0, +, _size_0);
      WL_JMP(FID_ir_powerCountFits);
    }
  }}

  WL_CASE(FID_ir_unionItems)
    Term _values_0 = r0;
    Term _acc_0 = r1;
    WL_OPEN
    if (term_aux(_values_0) == CID_Nil) {
      r0 = _acc_0;
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_1;
        STK(1) = FID_ir_unionItems_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_ir_unionItems_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_1;
        WL_CONT = term_tsk(FID_ir_unionItems_k1, _t_0);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_setBinary)) {
        u64 _t_1 = task_node(e, FID_ir_setBinary, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = 1ull;
        e.mem[_t_1 + 1] = _acc_0;
        e.mem[_t_1 + 2] = _f_0;
        return term_tsk(FID_ir_setBinary, _t_1);
      }
      r0 = 1ull;
      r1 = _acc_0;
      r2 = _f_0;
      WL_JMP(FID_ir_setBinary);
    }
  }}

  WL_CASE(FID_ir_unionItems_k1)
    WL_POPN(1);
    Term _f_2 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    if (term_aux(_h_0) == CID_ir_VE) {
      Term _fb_1[1];
      u64 _sp_1 = ctr_take(e, _h_0, 1, _fb_1);
      Term _f_3 = _fb_1[0];
      term_sink(e, _f_2);
      u64 _nd_0 = _sp_1 >= HEAP_OFF ? _sp_1 : heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_3);
      r0 = term_ctr(CID_ir_VE, _nd_0);
      WL_RETN(1);
    } else {
      if (!DEVICE && !seq && fid_nofk(FID_ir_unionItems)) {
        u64 _t_2 = task_node(e, FID_ir_unionItems, WL_CONT, WL_IDX, 0);
        e.mem[_t_2 + 0] = _f_2;
        e.mem[_t_2 + 1] = _h_0;
        return term_tsk(FID_ir_unionItems, _t_2);
      }
      r0 = _f_2;
      r1 = _h_0;
      WL_JMP(FID_ir_unionItems);
    }
  }}

  WL_CASE(FID_ir_domainIndex)
    Term _domain_0 = r0;
    Term _value_0 = r1;
    WL_OPEN
    if (term_aux(_domain_0) == CID_ir_IntDomain) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _domain_0, 2, _fb_0);
      u32 _f_0 = _fb_0[0];
      u32 _f_1 = _fb_0[1];
      if (term_aux(_value_0) == CID_ir_VI) {
        u32 _bits_0 = term_loc(_value_0);
        Term _o_0[2];
        if (spin_ir_domainRangeIndex(e, _o_0, U32_BIN(_bits_0, -, _f_0), _f_1) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(2), _sp_0);
        r0 = (u32)_o_0[0];
        r1 = (u32)_o_0[1];
        WL_RETN(2);
      } else {
        term_sink(e, _value_0);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = 0;
        r1 = 0;
        WL_RETN(2);
      }
    } else {
      Term _fb_1[1];
      u64 _sp_1 = ctr_take(e, _domain_0, 1, _fb_1);
      Term _f_2 = _fb_1[0];
      spare_free(e, cls_fit(1), _sp_1);
      if (!DEVICE && !seq && fid_nofk(FID_ir_domainTreeIndex)) {
        u64 _t_0 = task_node(e, FID_ir_domainTreeIndex, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _f_2;
        e.mem[_t_0 + 1] = _value_0;
        e.mem[_t_0 + 2] = 0ull;
        return term_tsk(FID_ir_domainTreeIndex, _t_0);
      }
      r0 = _f_2;
      r1 = _value_0;
      r2 = 0ull;
      WL_JMP(FID_ir_domainTreeIndex);
    }
  }}

  WL_CASE(FID_List_sort_0)
    Term _xs_0 = r0;
    WL_OPEN
    if (term_aux(_xs_0) == CID_Nil) {
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, _f_1);
      Term _ys_0 = term_ctr(CID_Con, _nd_0);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _ys_0;
        STK(1) = FID_List_sort_0_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_List_sort_0_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _ys_0;
        WL_CONT = term_tsk(FID_List_sort_0_k1, _t_0);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_List_length)) {
        u64 _t_1 = task_node(e, FID_List_length, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _ys_0;
        return term_tsk(FID_List_length, _t_1);
      }
      r0 = _ys_0;
      WL_JMP(FID_List_length);
    }
  }}

  WL_CASE(FID_List_sort_0_k1)
    WL_POPN(1);
    Term _ys_1 = STK(0);
    Term _n_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _n_0;
      STK(1) = FID_List_sort_0_k3;
      WL_PUSHN(2);
    } else {
      u64 _t_2 = task_node(e, FID_List_sort_0_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _n_0;
      WL_CONT = term_tsk(FID_List_sort_0_k3, _t_2);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_List_sort_pairs_0)) {
      u64 _t_3 = task_node(e, FID_List_sort_pairs_0, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _ys_1;
      return term_tsk(FID_List_sort_pairs_0, _t_3);
    }
    r0 = _ys_1;
    WL_JMP(FID_List_sort_pairs_0);
  }}

  WL_CASE(FID_List_sort_0_k3)
    WL_POPN(1);
    Term _n_1 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_List_sort_go_0)) {
      u64 _t_4 = task_node(e, FID_List_sort_go_0, WL_CONT, WL_IDX, 0);
      e.mem[_t_4 + 0] = _n_1;
      e.mem[_t_4 + 1] = _n_1;
      e.mem[_t_4 + 2] = _h_0;
      return term_tsk(FID_List_sort_go_0, _t_4);
    }
    r0 = _n_1;
    r1 = _n_1;
    r2 = _h_0;
    WL_JMP(FID_List_sort_go_0);
  }}

  WL_CASE(FID_ir_setUnique)
    Term _values_0 = r0;
    u32 _previous_0 = r1;
    Term _previous_1 = r2;
    Term _reversed_0 = r3;
    u32 _size_0 = r4;
    WL_OPEN
    WL_SPIN
    if (term_aux(_values_0) == CID_Nil) {
      term_sink(e, _previous_1);
      Term _o_0[1];
      if (spin_List_reverse(e, _o_0, _reversed_0) == 0) {
        return 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
        u64 _t_0 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _o_0[0];
        e.mem[_t_0 + 1] = _size_0;
        return term_tsk(FID_ir_setSorted, _t_0);
      }
      r0 = _o_0[0];
      r1 = _size_0;
      WL_JMP(FID_ir_setSorted);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _values_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (_previous_0 == 0) {
        _f_0 = term_keep(e, _f_0, 1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        e.mem[_nd_0 + 1] = rfc_seal(e, _reversed_0);
        r0 = _f_1;
        r1 = 1;
        r2 = _f_0;
        r3 = term_ctr(CID_Con, _nd_0);
        r4 = U32_BIN(_size_0, +, 1);
        _values_0 = r0;
        _previous_0 = r1;
        _previous_1 = r2;
        _reversed_0 = r3;
        _size_0 = r4;
        WL_AGAIN(FID_ir_setUnique);
      } else {
        _f_0 = term_keep(e, _f_0, 1);
        spare_free(e, cls_fit(2), _sp_0);
        if (seq) {
          WL_ROOM(5);
          STK(0) = _f_0;
          STK(1) = _f_1;
          STK(2) = _reversed_0;
          STK(3) = _size_0;
          STK(4) = FID_ir_setUnique_k1;
          WL_PUSHN(5);
        } else {
          u64 _t_1 = task_node(e, FID_ir_setUnique_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_1 + 0] = _f_0;
          e.mem[_t_1 + 1] = _f_1;
          e.mem[_t_1 + 2] = _reversed_0;
          e.mem[_t_1 + 3] = _size_0;
          WL_CONT = term_tsk(FID_ir_setUnique_k1, _t_1);
          WL_IDX = 4;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_veq)) {
          u64 _t_2 = task_node(e, FID_ir_veq, WL_CONT, WL_IDX, 0);
          e.mem[_t_2 + 0] = _previous_1;
          e.mem[_t_2 + 1] = _f_0;
          return term_tsk(FID_ir_veq, _t_2);
        }
        r0 = _previous_1;
        r1 = _f_0;
        WL_JMP(FID_ir_veq);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_setUnique_k1)
    WL_POPN(4);
    Term _f_2 = STK(0);
    Term _f_3 = STK(1);
    Term _reversed_1 = STK(2);
    u32 _size_1 = STK(3);
    u32 _h_0 = r0;
    WL_OPEN
    if (_h_0 == 1) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setUnique)) {
        u64 _t_3 = task_node(e, FID_ir_setUnique, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _f_3;
        e.mem[_t_3 + 1] = 1;
        e.mem[_t_3 + 2] = _f_2;
        e.mem[_t_3 + 3] = _reversed_1;
        e.mem[_t_3 + 4] = _size_1;
        return term_tsk(FID_ir_setUnique, _t_3);
      }
      r0 = _f_3;
      r1 = 1;
      r2 = _f_2;
      r3 = _reversed_1;
      r4 = _size_1;
      WL_JMP(FID_ir_setUnique);
    } else {
      _f_2 = term_keep(e, _f_2, 1);
      u64 _nd_1 = heap_alloc(e, 1);
      e.mem[_nd_1 + 0] = rfc_seal(e, _f_2);
      e.mem[_nd_1 + 1] = rfc_seal(e, _reversed_1);
      if (!DEVICE && !seq && fid_nofk(FID_ir_setUnique)) {
        u64 _t_4 = task_node(e, FID_ir_setUnique, WL_CONT, WL_IDX, 0);
        e.mem[_t_4 + 0] = _f_3;
        e.mem[_t_4 + 1] = 1;
        e.mem[_t_4 + 2] = _f_2;
        e.mem[_t_4 + 3] = term_ctr(CID_Con, _nd_1);
        e.mem[_t_4 + 4] = U32_BIN(_size_1, +, 1);
        return term_tsk(FID_ir_setUnique, _t_4);
      }
      r0 = _f_3;
      r1 = 1;
      r2 = _f_2;
      r3 = term_ctr(CID_Con, _nd_1);
      r4 = U32_BIN(_size_1, +, 1);
      WL_JMP(FID_ir_setUnique);
    }
  }}

  WL_CASE(FID_ir_veq)
    Term _a_0 = r0;
    Term _b_0 = r1;
    WL_OPEN
    if (term_aux(_a_0) == CID_ir_VI) {
      u32 _x_0 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VI) {
        u32 _y_0 = term_loc(_b_0);
        r0 = U32_BIN(_x_0, ==, _y_0);
        WL_RETN(1);
      } else {
        term_sink(e, _b_0);
        r0 = 0;
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VB) {
      u32 _x_1 = term_loc(_a_0);
      if (term_aux(_b_0) == CID_ir_VB) {
        u32 _y_1 = term_loc(_b_0);
        Term _o_0[1];
        if (spin_ir_boolEq(e, _o_0, _x_1, _y_1) == 0) {
          return 0;
        }
        r0 = (u32)_o_0[0];
        WL_RETN(1);
      } else {
        term_sink(e, _b_0);
        r0 = 0;
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VS) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_b_0) == CID_ir_VS) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _b_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        Term _o_1[1];
        if (spin_String_eq(e, _o_1, _f_0, _f_1) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(1), _sp_1);
        spare_free(e, cls_fit(1), _sp_0);
        r0 = (u32)_o_1[0];
        WL_RETN(1);
      } else {
        term_sink(e, _f_0);
        term_sink(e, _b_0);
        spare_free(e, cls_fit(1), _sp_0);
        r0 = 0;
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VSet) {
      Term _fb_2[2];
      u64 _sp_2 = ctr_take(e, _a_0, 2, _fb_2);
      Term _f_2 = _fb_2[0];
      Term _f_3 = _fb_2[1];
      if (term_aux(_b_0) == CID_ir_VSet) {
        Term _fb_3[2];
        u64 _sp_3 = ctr_take(e, _b_0, 2, _fb_3);
        Term _f_4 = _fb_3[0];
        Term _f_5 = _fb_3[1];
        spare_free(e, cls_fit(2), _sp_3);
        spare_free(e, cls_fit(2), _sp_2);
        if (seq) {
          WL_ROOM(3);
          STK(0) = _f_3;
          STK(1) = _f_5;
          STK(2) = FID_ir_veq_k1;
          WL_PUSHN(3);
        } else {
          u64 _t_0 = task_node(e, FID_ir_veq_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _f_3;
          e.mem[_t_0 + 1] = _f_5;
          WL_CONT = term_tsk(FID_ir_veq_k1, _t_0);
          WL_IDX = 2;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_listEq)) {
          u64 _t_1 = task_node(e, FID_ir_listEq, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _f_2;
          e.mem[_t_1 + 1] = _f_4;
          return term_tsk(FID_ir_listEq, _t_1);
        }
        r0 = _f_2;
        r1 = _f_4;
        WL_JMP(FID_ir_listEq);
      } else {
        term_sink(e, _f_2);
        term_sink(e, _f_3);
        term_sink(e, _b_0);
        spare_free(e, cls_fit(2), _sp_2);
        r0 = 0;
        WL_RETN(1);
      }
    } else if (term_aux(_a_0) == CID_ir_VFun) {
      Term _fb_4[2];
      u64 _sp_4 = ctr_take(e, _a_0, 2, _fb_4);
      Term _f_8 = _fb_4[0];
      u32 _f_9 = _fb_4[1];
      if (term_aux(_b_0) == CID_ir_VFun) {
        Term _fb_5[2];
        u64 _sp_5 = ctr_take(e, _b_0, 2, _fb_5);
        Term _f_10 = _fb_5[0];
        u32 _f_11 = _fb_5[1];
        spare_free(e, cls_fit(2), _sp_5);
        spare_free(e, cls_fit(2), _sp_4);
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_veq_k3;
          WL_PUSHN(1);
        } else {
          u64 _t_2 = task_node(e, FID_ir_veq_k3, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_veq_k3, _t_2);
          WL_IDX = 0;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_functionOrder)) {
          u64 _t_3 = task_node(e, FID_ir_functionOrder, WL_CONT, WL_IDX, 0);
          e.mem[_t_3 + 0] = _f_8;
          e.mem[_t_3 + 1] = _f_10;
          return term_tsk(FID_ir_functionOrder, _t_3);
        }
        r0 = _f_8;
        r1 = _f_10;
        WL_JMP(FID_ir_functionOrder);
      } else {
        term_sink(e, _f_8);
        term_sink(e, _b_0);
        spare_free(e, cls_fit(2), _sp_4);
        r0 = 0;
        WL_RETN(1);
      }
    } else {
      term_sink(e, _a_0);
      term_sink(e, _b_0);
      r0 = 0;
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_veq_k1)
    WL_POPN(2);
    Term _f_6 = STK(0);
    Term _f_7 = STK(1);
    u32 _h_0 = r0;
    WL_OPEN
    Term _o_2[1];
    if (spin_ir_domainSize(e, _o_2, _f_6) == 0) {
      return 0;
    }
    Term _o_3[1];
    if (spin_ir_domainSize(e, _o_3, _f_7) == 0) {
      return 0;
    }
    Term _o_4[1];
    if (spin_Bool_and(e, _o_4, U32_BIN((u32)_o_2[0], ==, (u32)_o_3[0]), _h_0) == 0) {
      return 0;
    }
    r0 = (u32)_o_4[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_veq_k3)
    u32 _h_1 = r0;
    WL_OPEN
    Term _o_5[1];
    if (spin_ir_functionEqual(e, _o_5, _h_1) == 0) {
      return 0;
    }
    r0 = (u32)_o_5[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionKeys)
    Term _tree_0 = r0;
    Term _acc_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      r0 = _acc_0;
      WL_RETN(1);
    } else {
      Term _fb_0[6];
      u64 _sp_0 = ctr_take(e, _tree_0, 6, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _f_2 = _fb_0[2];
      Term _f_3 = _fb_0[3];
      u32 _f_4 = _fb_0[4];
      u32 _f_5 = _fb_0[5];
      term_sink(e, _f_2);
      spare_free(e, cls_fit(6), _sp_0);
      if (seq) {
        WL_ROOM(3);
        STK(0) = _f_0;
        STK(1) = _f_1;
        STK(2) = FID_ir_functionKeys_k1;
        WL_PUSHN(3);
      } else {
        u64 _t_0 = task_node(e, FID_ir_functionKeys_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        e.mem[_t_0 + 1] = _f_1;
        WL_CONT = term_tsk(FID_ir_functionKeys_k1, _t_0);
        WL_IDX = 2;
      }
      r0 = _f_3;
      r1 = _acc_0;
      _tree_0 = r0;
      _acc_0 = r1;
      WL_AGAIN(FID_ir_functionKeys);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_functionKeys_k1)
    WL_POPN(2);
    Term _f_6 = STK(0);
    Term _f_7 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _f_7);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_0);
    if (!DEVICE && !seq && fid_nofk(FID_ir_functionKeys)) {
      u64 _t_1 = task_node(e, FID_ir_functionKeys, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _f_6;
      e.mem[_t_1 + 1] = term_ctr(CID_Con, _nd_0);
      return term_tsk(FID_ir_functionKeys, _t_1);
    }
    r0 = _f_6;
    r1 = term_ctr(CID_Con, _nd_0);
    WL_JMP(FID_ir_functionKeys);
  }}

  WL_CASE(FID_ir_setSorted)
    Term _items_0 = r0;
    u32 _size_0 = r1;
    WL_OPEN
    u32 _s_0 = U32_BIN(_size_0, <=, 1048576ull);
    if (_s_0 == 0) {
      term_sink(e, _items_0);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 948);
      WL_RETN(1);
    } else {
      _items_0 = term_keep(e, _items_0, 1);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _items_0;
        STK(1) = FID_ir_setSorted_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_ir_setSorted_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _items_0;
        WL_CONT = term_tsk(FID_ir_setSorted_k1, _t_0);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_domainSorted)) {
        u64 _t_1 = task_node(e, FID_ir_domainSorted, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _items_0;
        e.mem[_t_1 + 1] = _size_0;
        return term_tsk(FID_ir_domainSorted, _t_1);
      }
      r0 = _items_0;
      r1 = _size_0;
      WL_JMP(FID_ir_domainSorted);
    }
  }}

  WL_CASE(FID_ir_setSorted_k1)
    WL_POPN(1);
    Term _items_1 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _items_1);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_0);
    r0 = term_ctr(CID_ir_VSet, _nd_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionFind)
    Term _tree_0 = r0;
    Term _key_0 = r1;
    WL_OPEN
    if (term_aux(_tree_0) == CID_ir_FunctionEmpty) {
      term_sink(e, _key_0);
      r0 = 0;
      r1 = 0;
      WL_RETN(2);
    } else {
      Term _fb_0[6];
      u64 _sp_0 = ctr_take(e, _tree_0, 6, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _f_2 = _fb_0[2];
      Term _f_3 = _fb_0[3];
      u32 _f_4 = _fb_0[4];
      u32 _f_5 = _fb_0[5];
      _key_0 = term_keep(e, _key_0, 1);
      spare_free(e, cls_fit(6), _sp_0);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _f_0;
        STK(1) = _f_2;
        STK(2) = _f_3;
        STK(3) = _key_0;
        STK(4) = FID_ir_functionFind_k1;
        WL_PUSHN(5);
      } else {
        u64 _t_0 = task_node(e, FID_ir_functionFind_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        e.mem[_t_0 + 1] = _f_2;
        e.mem[_t_0 + 2] = _f_3;
        e.mem[_t_0 + 3] = _key_0;
        WL_CONT = term_tsk(FID_ir_functionFind_k1, _t_0);
        WL_IDX = 4;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_valueOrder)) {
        u64 _t_1 = task_node(e, FID_ir_valueOrder, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _key_0;
        e.mem[_t_1 + 1] = _f_1;
        return term_tsk(FID_ir_valueOrder, _t_1);
      }
      r0 = _key_0;
      r1 = _f_1;
      WL_JMP(FID_ir_valueOrder);
    }
  }}

  WL_CASE(FID_ir_functionFind_k1)
    WL_POPN(4);
    Term _f_6 = STK(0);
    Term _f_7 = STK(1);
    Term _f_8 = STK(2);
    Term _key_1 = STK(3);
    u32 _h_0 = r0;
    WL_OPEN
    if (_h_0 == 1) {
      term_sink(e, _f_6);
      term_sink(e, _f_8);
      term_sink(e, _key_1);
      r0 = 1;
      r1 = _f_7;
      WL_RETN(2);
    } else if (_h_0 == 0) {
      term_sink(e, _f_7);
      term_sink(e, _f_8);
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionFind)) {
        u64 _t_2 = task_node(e, FID_ir_functionFind, WL_CONT, WL_IDX, 0);
        e.mem[_t_2 + 0] = _f_6;
        e.mem[_t_2 + 1] = _key_1;
        return term_tsk(FID_ir_functionFind, _t_2);
      }
      r0 = _f_6;
      r1 = _key_1;
      WL_JMP(FID_ir_functionFind);
    } else {
      term_sink(e, _f_6);
      term_sink(e, _f_7);
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionFind)) {
        u64 _t_3 = task_node(e, FID_ir_functionFind, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _f_8;
        e.mem[_t_3 + 1] = _key_1;
        return term_tsk(FID_ir_functionFind, _t_3);
      }
      r0 = _f_8;
      r1 = _key_1;
      WL_JMP(FID_ir_functionFind);
    }
  }}

  WL_CASE(FID_ir_productCount)
    Term _domains_0 = r0;
    u32 _count_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_domains_0) == CID_Nil) {
      Term _o_0[1];
      if (spin_ir_productCountEnd(e, _o_0, U32_BIN(_count_0, <=, 1048576ull), _count_0) == 0) {
        return 0;
      }
      r0 = _o_0[0];
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _domains_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _f_0, 1, _fb_1);
        Term _f_2 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_1 >= HEAP_OFF ? _sp_1 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_2);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = term_ctr(CID_ir_VE, _nd_0);
        WL_RETN(1);
      } else if (term_aux(_f_0) == CID_ir_VSet) {
        Term _fb_2[2];
        u64 _sp_2 = ctr_take(e, _f_0, 2, _fb_2);
        Term _f_3 = _fb_2[0];
        Term _f_4 = _fb_2[1];
        term_sink(e, _f_3);
        Term _o_1[1];
        if (spin_ir_domainSize(e, _o_1, _f_4) == 0) {
          return 0;
        }
        if ((u32)_o_1[0] == 0) {
          spare_free(e, cls_fit(2), _sp_2);
          spare_free(e, cls_fit(2), _sp_0);
          r0 = _f_1;
          r1 = 0ull;
          _domains_0 = r0;
          _count_0 = r1;
          WL_AGAIN(FID_ir_productCount);
        } else {
          Term _a_0 = ((u32)((u32)_o_1[0]) == 0 ? 0 : (u64)U32_QUO((u32)(1048576ull), (u32)((u32)_o_1[0])));
          spare_free(e, cls_fit(2), _sp_2);
          spare_free(e, cls_fit(2), _sp_0);
          if (!DEVICE && !seq && fid_nofk(FID_ir_productCountFits)) {
            u64 _t_0 = task_node(e, FID_ir_productCountFits, WL_CONT, WL_IDX, 0);
            e.mem[_t_0 + 0] = U32_BIN(_count_0, <=, _a_0);
            e.mem[_t_0 + 1] = _f_1;
            e.mem[_t_0 + 2] = _count_0;
            e.mem[_t_0 + 3] = (u32)_o_1[0];
            return term_tsk(FID_ir_productCountFits, _t_0);
          }
          r0 = U32_BIN(_count_0, <=, _a_0);
          r1 = _f_1;
          r2 = _count_0;
          r3 = (u32)_o_1[0];
          WL_JMP(FID_ir_productCountFits);
        }
      } else {
        term_sink(e, _f_0);
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1339);
        WL_RETN(1);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_firstErr)
    Term _xs_0 = r0;
    WL_OPEN
    WL_SPIN
    if (term_aux(_xs_0) == CID_Nil) {
      r0 = 0;
      r1 = 0;
      WL_RETN(2);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _xs_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      if (term_aux(_f_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _f_0, 1, _fb_1);
        Term _f_2 = _fb_1[0];
        term_sink(e, _f_1);
        spare_free(e, cls_fit(1), _sp_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = 1;
        r1 = _f_2;
        WL_RETN(2);
      } else {
        term_sink(e, _f_0);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_1;
        _xs_0 = r0;
        WL_AGAIN(FID_ir_firstErr);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_boundedLoop)
    Term _nodes_0 = r0;
    Term _env_0 = r1;
    Term _env_1 = r2;
    u32 _op_0 = r3;
    u32 _slot_0 = r4;
    u32 _body_0 = r5;
    Term _remaining_0 = r6;
    Term _acc_0 = r7;
    WL_OPEN
    if (term_aux(_remaining_0) == CID_Nil) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      if (_op_0 == 1) {
        term_sink(e, _acc_0);
        r0 = term_pak(CID_ir_VB, 1);
        WL_RETN(1);
      } else if ((_op_0 & 3) == 1) {
        term_sink(e, _acc_0);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1026);
        WL_RETN(1);
      } else if (_op_0 == 3) {
        if (!DEVICE && !seq && fid_nofk(FID_ir_filteredFinish)) {
          u64 _t_0 = task_node(e, FID_ir_filteredFinish, WL_CONT, WL_IDX, 0);
          e.mem[_t_0 + 0] = _acc_0;
          e.mem[_t_0 + 1] = term_pak(CID_Nil, 0);
          e.mem[_t_0 + 2] = 0ull;
          return term_tsk(FID_ir_filteredFinish, _t_0);
        }
        r0 = _acc_0;
        r1 = term_pak(CID_Nil, 0);
        r2 = 0ull;
        WL_JMP(FID_ir_filteredFinish);
      } else if ((_op_0 & 3) == 3) {
        term_sink(e, _acc_0);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1026);
        WL_RETN(1);
      } else if (_op_0 == 2) {
        term_sink(e, _acc_0);
        r0 = term_pak(CID_ir_VB, 0);
        WL_RETN(1);
      } else if ((_op_0 & 3) == 2) {
        term_sink(e, _acc_0);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1026);
        WL_RETN(1);
      } else if (_op_0 == 4) {
        if (!DEVICE && !seq && fid_nofk(FID_ir_makeSet)) {
          u64 _t_1 = task_node(e, FID_ir_makeSet, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _acc_0;
          return term_tsk(FID_ir_makeSet, _t_1);
        }
        r0 = _acc_0;
        WL_JMP(FID_ir_makeSet);
      } else {
        term_sink(e, _acc_0);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1026);
        WL_RETN(1);
      }
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _remaining_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      _f_0 = term_keep(e, _f_0, 1);
      Term _o_0[2];
      if (spin_ir_bindLocal(e, _o_0, _env_0, _env_1, _slot_0, _f_0) == 0) {
        return 0;
      }
      _nodes_0 = term_keep(e, _nodes_0, 1);
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(10);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _op_0;
        STK(4) = _slot_0;
        STK(5) = _body_0;
        STK(6) = _f_0;
        STK(7) = _f_1;
        STK(8) = _acc_0;
        STK(9) = FID_ir_boundedLoop_k1;
        WL_PUSHN(10);
      } else {
        u64 _t_2 = task_node(e, FID_ir_boundedLoop_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_2 + 0] = _nodes_0;
        e.mem[_t_2 + 1] = _env_0;
        e.mem[_t_2 + 2] = _env_1;
        e.mem[_t_2 + 3] = _op_0;
        e.mem[_t_2 + 4] = _slot_0;
        e.mem[_t_2 + 5] = _body_0;
        e.mem[_t_2 + 6] = _f_0;
        e.mem[_t_2 + 7] = _f_1;
        e.mem[_t_2 + 8] = _acc_0;
        WL_CONT = term_tsk(FID_ir_boundedLoop_k1, _t_2);
        WL_IDX = 9;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
        u64 _t_3 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _nodes_0;
        e.mem[_t_3 + 1] = _o_0[0];
        e.mem[_t_3 + 2] = _o_0[1];
        e.mem[_t_3 + 3] = _body_0;
        return term_tsk(FID_ir_eval, _t_3);
      }
      r0 = _nodes_0;
      r1 = _o_0[0];
      r2 = _o_0[1];
      r3 = _body_0;
      WL_JMP(FID_ir_eval);
    }
  }}

  WL_CASE(FID_ir_boundedLoop_k1)
    WL_POPN(9);
    Term _nodes_1 = STK(0);
    Term _env_2 = STK(1);
    Term _env_3 = STK(2);
    u32 _op_1 = STK(3);
    u32 _slot_1 = STK(4);
    u32 _body_1 = STK(5);
    Term _f_2 = STK(6);
    Term _f_3 = STK(7);
    Term _acc_1 = STK(8);
    Term _h_0 = r0;
    WL_OPEN
    if (term_aux(_h_0) == CID_ir_VE) {
      term_sink(e, _nodes_1);
      term_sink(e, _env_2);
      term_sink(e, _env_3);
      term_sink(e, _f_2);
      term_sink(e, _f_3);
      term_sink(e, _acc_1);
      Term _fb_1[1];
      u64 _sp_1 = ctr_take(e, _h_0, 1, _fb_1);
      Term _f_4 = _fb_1[0];
      u64 _nd_0 = _sp_1 >= HEAP_OFF ? _sp_1 : heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_4);
      r0 = term_ctr(CID_ir_VE, _nd_0);
      WL_RETN(1);
    } else {
      if (_op_1 == 4) {
        term_sink(e, _f_2);
        u64 _nd_1 = heap_alloc(e, 1);
        e.mem[_nd_1 + 0] = rfc_seal(e, _h_0);
        e.mem[_nd_1 + 1] = rfc_seal(e, _acc_1);
        if (!DEVICE && !seq && fid_nofk(FID_ir_boundedLoop)) {
          u64 _t_4 = task_node(e, FID_ir_boundedLoop, WL_CONT, WL_IDX, 0);
          e.mem[_t_4 + 0] = _nodes_1;
          e.mem[_t_4 + 1] = _env_2;
          e.mem[_t_4 + 2] = _env_3;
          e.mem[_t_4 + 3] = 4ull;
          e.mem[_t_4 + 4] = _slot_1;
          e.mem[_t_4 + 5] = _body_1;
          e.mem[_t_4 + 6] = _f_3;
          e.mem[_t_4 + 7] = term_ctr(CID_Con, _nd_1);
          return term_tsk(FID_ir_boundedLoop, _t_4);
        }
        r0 = _nodes_1;
        r1 = _env_2;
        r2 = _env_3;
        r3 = 4ull;
        r4 = _slot_1;
        r5 = _body_1;
        r6 = _f_3;
        r7 = term_ctr(CID_Con, _nd_1);
        WL_JMP(FID_ir_boundedLoop);
      } else {
        if (!DEVICE && !seq && fid_nofk(FID_ir_boundedBoolean)) {
          u64 _t_5 = task_node(e, FID_ir_boundedBoolean, WL_CONT, WL_IDX, 0);
          e.mem[_t_5 + 0] = _nodes_1;
          e.mem[_t_5 + 1] = _env_2;
          e.mem[_t_5 + 2] = _env_3;
          e.mem[_t_5 + 3] = _op_1;
          e.mem[_t_5 + 4] = _slot_1;
          e.mem[_t_5 + 5] = _body_1;
          e.mem[_t_5 + 6] = _f_2;
          e.mem[_t_5 + 7] = _f_3;
          e.mem[_t_5 + 8] = _acc_1;
          e.mem[_t_5 + 9] = _h_0;
          return term_tsk(FID_ir_boundedBoolean, _t_5);
        }
        r0 = _nodes_1;
        r1 = _env_2;
        r2 = _env_3;
        r3 = _op_1;
        r4 = _slot_1;
        r5 = _body_1;
        r6 = _f_2;
        r7 = _f_3;
        r8 = _acc_1;
        r9 = _h_0;
        WL_JMP(FID_ir_boundedBoolean);
      }
    }
  }}

  WL_CASE(FID_ir_setBinaryValues)
    u32 _op_0 = r0;
    Term _left_0 = r1;
    Term _right_0 = r2;
    WL_OPEN
    if (term_aux(_left_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _left_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_right_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _right_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        spare_free(e, cls_fit(1), _sp_1);
        r0 = term_ctr(CID_ir_VE, _nd_0);
        WL_RETN(1);
      } else {
        term_sink(e, _right_0);
        u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
        r0 = term_ctr(CID_ir_VE, _nd_1);
        WL_RETN(1);
      }
    } else if (term_aux(_left_0) == CID_ir_VSet) {
      Term _fb_2[2];
      u64 _sp_2 = ctr_take(e, _left_0, 2, _fb_2);
      Term _f_2 = _fb_2[0];
      Term _f_3 = _fb_2[1];
      if (term_aux(_right_0) == CID_ir_VE) {
        term_sink(e, _f_2);
        term_sink(e, _f_3);
        Term _fb_3[1];
        u64 _sp_3 = ctr_take(e, _right_0, 1, _fb_3);
        Term _f_4 = _fb_3[0];
        u64 _nd_2 = _sp_3 >= HEAP_OFF ? _sp_3 : heap_alloc(e, 0);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_4);
        spare_free(e, cls_fit(2), _sp_2);
        r0 = term_ctr(CID_ir_VE, _nd_2);
        WL_RETN(1);
      } else if (term_aux(_right_0) == CID_ir_VSet) {
        Term _fb_4[2];
        u64 _sp_4 = ctr_take(e, _right_0, 2, _fb_4);
        Term _f_5 = _fb_4[0];
        Term _f_6 = _fb_4[1];
        if (_op_0 == 1) {
          if (term_aux(_f_2) == CID_Nil) {
            term_sink(e, _f_3);
            if (term_aux(_f_5) == CID_Nil) {
              u64 _nd_3 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_3 + 0] = rfc_seal(e, term_pak(CID_Nil, 0));
              e.mem[_nd_3 + 1] = rfc_seal(e, _f_6);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_3);
              WL_RETN(1);
            } else {
              u64 _nd_4 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_4 + 0] = rfc_seal(e, _f_5);
              e.mem[_nd_4 + 1] = rfc_seal(e, _f_6);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_4);
              WL_RETN(1);
            }
          } else {
            if (term_aux(_f_5) == CID_Nil) {
              term_sink(e, _f_6);
              u64 _nd_5 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_5 + 0] = rfc_seal(e, _f_2);
              e.mem[_nd_5 + 1] = rfc_seal(e, _f_3);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_5);
              WL_RETN(1);
            } else {
              term_sink(e, _f_3);
              term_sink(e, _f_6);
              spare_free(e, cls_fit(2), _sp_4);
              spare_free(e, cls_fit(2), _sp_2);
              if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryLists)) {
                u64 _t_0 = task_node(e, FID_ir_setBinaryLists, WL_CONT, WL_IDX, 0);
                e.mem[_t_0 + 0] = 1ull;
                e.mem[_t_0 + 1] = _f_2;
                e.mem[_t_0 + 2] = _f_5;
                return term_tsk(FID_ir_setBinaryLists, _t_0);
              }
              r0 = 1ull;
              r1 = _f_2;
              r2 = _f_5;
              WL_JMP(FID_ir_setBinaryLists);
            }
          }
        } else if ((_op_0 & 3) == 1) {
          term_sink(e, _f_3);
          term_sink(e, _f_6);
          spare_free(e, cls_fit(2), _sp_4);
          spare_free(e, cls_fit(2), _sp_2);
          if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryLists)) {
            u64 _t_1 = task_node(e, FID_ir_setBinaryLists, WL_CONT, WL_IDX, 0);
            e.mem[_t_1 + 0] = _op_0;
            e.mem[_t_1 + 1] = _f_2;
            e.mem[_t_1 + 2] = _f_5;
            return term_tsk(FID_ir_setBinaryLists, _t_1);
          }
          r0 = _op_0;
          r1 = _f_2;
          r2 = _f_5;
          WL_JMP(FID_ir_setBinaryLists);
        } else if (_op_0 == 3) {
          if (term_aux(_f_2) == CID_Nil) {
            if (term_aux(_f_5) == CID_Nil) {
              term_sink(e, _f_6);
              u64 _nd_6 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_6 + 0] = rfc_seal(e, term_pak(CID_Nil, 0));
              e.mem[_nd_6 + 1] = rfc_seal(e, _f_3);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_6);
              WL_RETN(1);
            } else {
              term_sink(e, _f_5);
              term_sink(e, _f_6);
              u64 _nd_7 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_7 + 0] = rfc_seal(e, term_pak(CID_Nil, 0));
              e.mem[_nd_7 + 1] = rfc_seal(e, _f_3);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_7);
              WL_RETN(1);
            }
          } else {
            if (term_aux(_f_5) == CID_Nil) {
              term_sink(e, _f_6);
              u64 _nd_8 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_8 + 0] = rfc_seal(e, _f_2);
              e.mem[_nd_8 + 1] = rfc_seal(e, _f_3);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_8);
              WL_RETN(1);
            } else {
              term_sink(e, _f_3);
              term_sink(e, _f_6);
              spare_free(e, cls_fit(2), _sp_4);
              spare_free(e, cls_fit(2), _sp_2);
              if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryLists)) {
                u64 _t_2 = task_node(e, FID_ir_setBinaryLists, WL_CONT, WL_IDX, 0);
                e.mem[_t_2 + 0] = 3ull;
                e.mem[_t_2 + 1] = _f_2;
                e.mem[_t_2 + 2] = _f_5;
                return term_tsk(FID_ir_setBinaryLists, _t_2);
              }
              r0 = 3ull;
              r1 = _f_2;
              r2 = _f_5;
              WL_JMP(FID_ir_setBinaryLists);
            }
          }
        } else if ((_op_0 & 3) == 3) {
          term_sink(e, _f_3);
          term_sink(e, _f_6);
          spare_free(e, cls_fit(2), _sp_4);
          spare_free(e, cls_fit(2), _sp_2);
          if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryLists)) {
            u64 _t_3 = task_node(e, FID_ir_setBinaryLists, WL_CONT, WL_IDX, 0);
            e.mem[_t_3 + 0] = _op_0;
            e.mem[_t_3 + 1] = _f_2;
            e.mem[_t_3 + 2] = _f_5;
            return term_tsk(FID_ir_setBinaryLists, _t_3);
          }
          r0 = _op_0;
          r1 = _f_2;
          r2 = _f_5;
          WL_JMP(FID_ir_setBinaryLists);
        } else if (_op_0 == 2) {
          if (term_aux(_f_2) == CID_Nil) {
            if (term_aux(_f_5) == CID_Nil) {
              term_sink(e, _f_6);
              u64 _nd_9 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_9 + 0] = rfc_seal(e, term_pak(CID_Nil, 0));
              e.mem[_nd_9 + 1] = rfc_seal(e, _f_3);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_9);
              WL_RETN(1);
            } else {
              term_sink(e, _f_5);
              term_sink(e, _f_6);
              u64 _nd_10 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_10 + 0] = rfc_seal(e, term_pak(CID_Nil, 0));
              e.mem[_nd_10 + 1] = rfc_seal(e, _f_3);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_10);
              WL_RETN(1);
            }
          } else {
            term_sink(e, _f_3);
            if (term_aux(_f_5) == CID_Nil) {
              term_sink(e, _f_2);
              u64 _nd_11 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 1);
              e.mem[_nd_11 + 0] = rfc_seal(e, term_pak(CID_Nil, 0));
              e.mem[_nd_11 + 1] = rfc_seal(e, _f_6);
              spare_free(e, cls_fit(2), _sp_4);
              r0 = term_ctr(CID_ir_VSet, _nd_11);
              WL_RETN(1);
            } else {
              term_sink(e, _f_6);
              spare_free(e, cls_fit(2), _sp_4);
              spare_free(e, cls_fit(2), _sp_2);
              if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryLists)) {
                u64 _t_4 = task_node(e, FID_ir_setBinaryLists, WL_CONT, WL_IDX, 0);
                e.mem[_t_4 + 0] = 2ull;
                e.mem[_t_4 + 1] = _f_2;
                e.mem[_t_4 + 2] = _f_5;
                return term_tsk(FID_ir_setBinaryLists, _t_4);
              }
              r0 = 2ull;
              r1 = _f_2;
              r2 = _f_5;
              WL_JMP(FID_ir_setBinaryLists);
            }
          }
        } else if ((_op_0 & 3) == 2) {
          term_sink(e, _f_3);
          term_sink(e, _f_6);
          spare_free(e, cls_fit(2), _sp_4);
          spare_free(e, cls_fit(2), _sp_2);
          if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryLists)) {
            u64 _t_5 = task_node(e, FID_ir_setBinaryLists, WL_CONT, WL_IDX, 0);
            e.mem[_t_5 + 0] = _op_0;
            e.mem[_t_5 + 1] = _f_2;
            e.mem[_t_5 + 2] = _f_5;
            return term_tsk(FID_ir_setBinaryLists, _t_5);
          }
          r0 = _op_0;
          r1 = _f_2;
          r2 = _f_5;
          WL_JMP(FID_ir_setBinaryLists);
        } else if (_op_0 == 4) {
          Term _o_0[1];
          if (spin_ir_domainSize(e, _o_0, _f_3) == 0) {
            return 0;
          }
          Term _o_1[1];
          if (spin_ir_domainSize(e, _o_1, _f_6) == 0) {
            return 0;
          }
          u32 _s_0 = U32_BIN((u32)_o_0[0], <=, (u32)_o_1[0]);
          if (_s_0 == 0) {
            term_sink(e, _f_2);
            term_sink(e, _f_5);
            spare_free(e, cls_fit(2), _sp_4);
            spare_free(e, cls_fit(2), _sp_2);
            r0 = term_pak(CID_ir_VB, 0);
            WL_RETN(1);
          } else {
            spare_free(e, cls_fit(2), _sp_4);
            spare_free(e, cls_fit(2), _sp_2);
            if (seq) {
              WL_ROOM(1);
              STK(0) = FID_ir_setBinaryValues_k1;
              WL_PUSHN(1);
            } else {
              u64 _t_6 = task_node(e, FID_ir_setBinaryValues_k1, WL_CONT, WL_IDX, 1);
              WL_CONT = term_tsk(FID_ir_setBinaryValues_k1, _t_6);
              WL_IDX = 0;
            }
            if (!DEVICE && !seq && fid_nofk(FID_ir_setSubset)) {
              u64 _t_7 = task_node(e, FID_ir_setSubset, WL_CONT, WL_IDX, 0);
              e.mem[_t_7 + 0] = _f_2;
              e.mem[_t_7 + 1] = _f_5;
              return term_tsk(FID_ir_setSubset, _t_7);
            }
            r0 = _f_2;
            r1 = _f_5;
            WL_JMP(FID_ir_setSubset);
          }
        } else {
          term_sink(e, _f_3);
          term_sink(e, _f_6);
          spare_free(e, cls_fit(2), _sp_4);
          spare_free(e, cls_fit(2), _sp_2);
          if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryLists)) {
            u64 _t_8 = task_node(e, FID_ir_setBinaryLists, WL_CONT, WL_IDX, 0);
            e.mem[_t_8 + 0] = _op_0;
            e.mem[_t_8 + 1] = _f_2;
            e.mem[_t_8 + 2] = _f_5;
            return term_tsk(FID_ir_setBinaryLists, _t_8);
          }
          r0 = _op_0;
          r1 = _f_2;
          r2 = _f_5;
          WL_JMP(FID_ir_setBinaryLists);
        }
      } else {
        term_sink(e, _f_2);
        term_sink(e, _f_3);
        term_sink(e, _right_0);
        spare_free(e, cls_fit(2), _sp_2);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1607);
        WL_RETN(1);
      }
    } else {
      term_sink(e, _left_0);
      if (term_aux(_right_0) == CID_ir_VE) {
        Term _fb_5[1];
        u64 _sp_5 = ctr_take(e, _right_0, 1, _fb_5);
        Term _f_7 = _fb_5[0];
        u64 _nd_12 = _sp_5 >= HEAP_OFF ? _sp_5 : heap_alloc(e, 0);
        e.mem[_nd_12 + 0] = rfc_seal(e, _f_7);
        r0 = term_ctr(CID_ir_VE, _nd_12);
        WL_RETN(1);
      } else {
        term_sink(e, _right_0);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1607);
        WL_RETN(1);
      }
    }
  }}

  WL_CASE(FID_ir_setBinaryValues_k1)
    u32 _h_0 = r0;
    WL_OPEN
    r0 = term_pak(CID_ir_VB, _h_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_valueMember)
    Term _value_0 = r0;
    Term _set_0 = r1;
    u32 _negate_0 = r2;
    WL_OPEN
    if (term_aux(_value_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _value_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_set_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _set_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        spare_free(e, cls_fit(1), _sp_1);
        r0 = term_ctr(CID_ir_VE, _nd_0);
        WL_RETN(1);
      } else if (term_aux(_set_0) == CID_ir_VSet) {
        Term _fb_2[2];
        u64 _sp_2 = ctr_take(e, _set_0, 2, _fb_2);
        Term _f_2 = _fb_2[0];
        Term _f_3 = _fb_2[1];
        term_sink(e, _f_2);
        term_sink(e, _f_3);
        u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
        spare_free(e, cls_fit(2), _sp_2);
        r0 = term_ctr(CID_ir_VE, _nd_1);
        WL_RETN(1);
      } else {
        term_sink(e, _set_0);
        u64 _nd_2 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_0);
        r0 = term_ctr(CID_ir_VE, _nd_2);
        WL_RETN(1);
      }
    } else {
      if (term_aux(_set_0) == CID_ir_VE) {
        term_sink(e, _value_0);
        Term _fb_3[1];
        u64 _sp_3 = ctr_take(e, _set_0, 1, _fb_3);
        Term _f_4 = _fb_3[0];
        u64 _nd_3 = _sp_3 >= HEAP_OFF ? _sp_3 : heap_alloc(e, 0);
        e.mem[_nd_3 + 0] = rfc_seal(e, _f_4);
        r0 = term_ctr(CID_ir_VE, _nd_3);
        WL_RETN(1);
      } else if (term_aux(_set_0) == CID_ir_VSet) {
        Term _fb_4[2];
        u64 _sp_4 = ctr_take(e, _set_0, 2, _fb_4);
        Term _f_5 = _fb_4[0];
        Term _f_6 = _fb_4[1];
        term_sink(e, _f_5);
        spare_free(e, cls_fit(2), _sp_4);
        if (seq) {
          WL_ROOM(2);
          STK(0) = _negate_0;
          STK(1) = FID_ir_valueMember_k1;
          WL_PUSHN(2);
        } else {
          u64 _t_0 = task_node(e, FID_ir_valueMember_k1, WL_CONT, WL_IDX, 1);
          e.mem[_t_0 + 0] = _negate_0;
          WL_CONT = term_tsk(FID_ir_valueMember_k1, _t_0);
          WL_IDX = 1;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_domainIndex)) {
          u64 _t_1 = task_node(e, FID_ir_domainIndex, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _f_6;
          e.mem[_t_1 + 1] = _value_0;
          return term_tsk(FID_ir_domainIndex, _t_1);
        }
        r0 = _f_6;
        r1 = _value_0;
        WL_JMP(FID_ir_domainIndex);
      } else {
        term_sink(e, _value_0);
        term_sink(e, _set_0);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1670);
        WL_RETN(1);
      }
    }
  }}

  WL_CASE(FID_ir_valueMember_k1)
    WL_POPN(1);
    u32 _negate_1 = STK(0);
    u32 _h_0 = r0;
    u32 _h_1 = r1;
    WL_OPEN
    Term _o_0[1];
    if (spin_ir_memberResult(e, _o_0, _h_0, _h_1, _negate_1) == 0) {
      return 0;
    }
    r0 = _o_0[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_localFind)
    Term _locals_0 = r0;
    u32 _slot_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_locals_0) == CID_Nil) {
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 1719);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _locals_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _fb_1[2];
      u64 _sp_1 = ctr_take(e, _f_0, 2, _fb_1);
      u32 _f_2 = _fb_1[0];
      Term _f_3 = _fb_1[1];
      spare_free(e, cls_fit(2), _sp_1);
      u32 _s_0 = U32_BIN(_f_2, ==, _slot_0);
      if (_s_0 == 1) {
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_3;
        WL_RETN(1);
      } else {
        term_sink(e, _f_3);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_1;
        r1 = _slot_0;
        _locals_0 = r0;
        _slot_0 = r1;
        WL_AGAIN(FID_ir_localFind);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_veqb)
    Term _a_0 = r0;
    Term _b_0 = r1;
    WL_OPEN
    if (term_aux(_a_0) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _a_0, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      if (term_aux(_b_0) == CID_ir_VE) {
        Term _fb_1[1];
        u64 _sp_1 = ctr_take(e, _b_0, 1, _fb_1);
        Term _f_1 = _fb_1[0];
        term_sink(e, _f_1);
        u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
        spare_free(e, cls_fit(1), _sp_1);
        r0 = term_ctr(CID_ir_VE, _nd_0);
        WL_RETN(1);
      } else {
        term_sink(e, _b_0);
        u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
        e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
        r0 = term_ctr(CID_ir_VE, _nd_1);
        WL_RETN(1);
      }
    } else {
      if (term_aux(_b_0) == CID_ir_VE) {
        term_sink(e, _a_0);
        Term _fb_2[1];
        u64 _sp_2 = ctr_take(e, _b_0, 1, _fb_2);
        Term _f_2 = _fb_2[0];
        u64 _nd_2 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 0);
        e.mem[_nd_2 + 0] = rfc_seal(e, _f_2);
        r0 = term_ctr(CID_ir_VE, _nd_2);
        WL_RETN(1);
      } else {
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_veqb_k1;
          WL_PUSHN(1);
        } else {
          u64 _t_0 = task_node(e, FID_ir_veqb_k1, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_veqb_k1, _t_0);
          WL_IDX = 0;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_veq)) {
          u64 _t_1 = task_node(e, FID_ir_veq, WL_CONT, WL_IDX, 0);
          e.mem[_t_1 + 0] = _a_0;
          e.mem[_t_1 + 1] = _b_0;
          return term_tsk(FID_ir_veq, _t_1);
        }
        r0 = _a_0;
        r1 = _b_0;
        WL_JMP(FID_ir_veq);
      }
    }
  }}

  WL_CASE(FID_ir_veqb_k1)
    u32 _h_0 = r0;
    WL_OPEN
    r0 = term_pak(CID_ir_VB, _h_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_functionSorted)
    Term _keys_0 = r0;
    Term _values_0 = r1;
    WL_OPEN
    _values_0 = term_keep(e, _values_0, 1);
    if (seq) {
      WL_ROOM(3);
      STK(0) = _keys_0;
      STK(1) = _values_0;
      STK(2) = FID_ir_functionSorted_k1;
      WL_PUSHN(3);
    } else {
      u64 _t_0 = task_node(e, FID_ir_functionSorted_k1, WL_CONT, WL_IDX, 1);
      e.mem[_t_0 + 0] = _keys_0;
      e.mem[_t_0 + 1] = _values_0;
      WL_CONT = term_tsk(FID_ir_functionSorted_k1, _t_0);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_firstErr)) {
      u64 _t_1 = task_node(e, FID_ir_firstErr, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _values_0;
      return term_tsk(FID_ir_firstErr, _t_1);
    }
    r0 = _values_0;
    WL_JMP(FID_ir_firstErr);
  }}

  WL_CASE(FID_ir_functionSorted_k1)
    WL_POPN(2);
    Term _keys_1 = STK(0);
    Term _values_1 = STK(1);
    u32 _h_0 = r0;
    Term _h_1 = r1;
    WL_OPEN
    if (_h_0 == 1) {
      term_sink(e, _keys_1);
      term_sink(e, _values_1);
      u64 _nd_0 = heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _h_1);
      r0 = term_ctr(CID_ir_VE, _nd_0);
      WL_RETN(1);
    } else {
      if (seq) {
        WL_ROOM(3);
        STK(0) = _keys_1;
        STK(1) = _values_1;
        STK(2) = FID_ir_functionSorted_k3;
        WL_PUSHN(3);
      } else {
        u64 _t_2 = task_node(e, FID_ir_functionSorted_k3, WL_CONT, WL_IDX, 1);
        e.mem[_t_2 + 0] = _keys_1;
        e.mem[_t_2 + 1] = _values_1;
        WL_CONT = term_tsk(FID_ir_functionSorted_k3, _t_2);
        WL_IDX = 2;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_vlen)) {
        u64 _t_3 = task_node(e, FID_ir_vlen, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _keys_1;
        return term_tsk(FID_ir_vlen, _t_3);
      }
      r0 = _keys_1;
      WL_JMP(FID_ir_vlen);
    }
  }}

  WL_CASE(FID_ir_functionSorted_k3)
    WL_POPN(2);
    Term _keys_2 = STK(0);
    Term _values_2 = STK(1);
    u32 _h_2 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _keys_2;
      STK(1) = _values_2;
      STK(2) = _h_2;
      STK(3) = FID_ir_functionSorted_k5;
      WL_PUSHN(4);
    } else {
      u64 _t_4 = task_node(e, FID_ir_functionSorted_k5, WL_CONT, WL_IDX, 1);
      e.mem[_t_4 + 0] = _keys_2;
      e.mem[_t_4 + 1] = _values_2;
      e.mem[_t_4 + 2] = _h_2;
      WL_CONT = term_tsk(FID_ir_functionSorted_k5, _t_4);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_vlen)) {
      u64 _t_5 = task_node(e, FID_ir_vlen, WL_CONT, WL_IDX, 0);
      e.mem[_t_5 + 0] = _values_2;
      return term_tsk(FID_ir_vlen, _t_5);
    }
    r0 = _values_2;
    WL_JMP(FID_ir_vlen);
  }}

  WL_CASE(FID_ir_functionSorted_k5)
    WL_POPN(3);
    Term _keys_3 = STK(0);
    Term _values_3 = STK(1);
    u32 _h_4 = STK(2);
    u32 _h_3 = r0;
    WL_OPEN
    u32 _s_0 = U32_BIN(_h_4, ==, _h_3);
    if (_s_0 == 0) {
      term_sink(e, _keys_3);
      term_sink(e, _values_3);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 228);
      WL_RETN(1);
    } else {
      u32 _s_1 = U32_BIN(_h_4, <=, 1048576ull);
      if (_s_1 == 0) {
        term_sink(e, _keys_3);
        term_sink(e, _values_3);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 393);
        WL_RETN(1);
      } else {
        _keys_3 = term_keep(e, _keys_3, 1);
        if (seq) {
          WL_ROOM(2);
          STK(0) = _keys_3;
          STK(1) = FID_ir_functionSorted_k7;
          WL_PUSHN(2);
        } else {
          u64 _t_6 = task_node(e, FID_ir_functionSorted_k7, WL_CONT, WL_IDX, 1);
          e.mem[_t_6 + 0] = _keys_3;
          WL_CONT = term_tsk(FID_ir_functionSorted_k7, _t_6);
          WL_IDX = 1;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_buildFunctionTree)) {
          u64 _t_7 = task_node(e, FID_ir_buildFunctionTree, WL_CONT, WL_IDX, 0);
          e.mem[_t_7 + 0] = _h_4;
          e.mem[_t_7 + 1] = _keys_3;
          e.mem[_t_7 + 2] = _values_3;
          return term_tsk(FID_ir_buildFunctionTree, _t_7);
        }
        r0 = _h_4;
        r1 = _keys_3;
        r2 = _values_3;
        WL_JMP(FID_ir_buildFunctionTree);
      }
    }
  }}

  WL_CASE(FID_ir_functionSorted_k7)
    WL_POPN(1);
    Term _keys_4 = STK(0);
    Term _h_5 = r0;
    Term _h_6 = r1;
    Term _h_7 = r2;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _h_5;
      STK(1) = _h_6;
      STK(2) = _h_7;
      STK(3) = FID_ir_functionSorted_k9;
      WL_PUSHN(4);
    } else {
      u64 _t_8 = task_node(e, FID_ir_functionSorted_k9, WL_CONT, WL_IDX, 1);
      e.mem[_t_8 + 0] = _h_5;
      e.mem[_t_8 + 1] = _h_6;
      e.mem[_t_8 + 2] = _h_7;
      WL_CONT = term_tsk(FID_ir_functionSorted_k9, _t_8);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_functionSequenceKeys)) {
      u64 _t_9 = task_node(e, FID_ir_functionSequenceKeys, WL_CONT, WL_IDX, 0);
      e.mem[_t_9 + 0] = _keys_4;
      e.mem[_t_9 + 1] = 1ull;
      return term_tsk(FID_ir_functionSequenceKeys, _t_9);
    }
    r0 = _keys_4;
    r1 = 1ull;
    WL_JMP(FID_ir_functionSequenceKeys);
  }}

  WL_CASE(FID_ir_functionSorted_k9)
    WL_POPN(3);
    Term _h_9 = STK(0);
    Term _h_10 = STK(1);
    Term _h_11 = STK(2);
    u32 _h_8 = r0;
    WL_OPEN
    Term _o_0[1];
    if (spin_ir_functionBuiltValue(e, _o_0, _h_9, _h_10, _h_11, _h_8) == 0) {
      return 0;
    }
    r0 = _o_0[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_setBinary)
    u32 _op_0 = r0;
    Term _left_0 = r1;
    Term _right_0 = r2;
    WL_OPEN
    if (_op_0 == 5) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_valueMember)) {
        u64 _t_0 = task_node(e, FID_ir_valueMember, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = _left_0;
        e.mem[_t_0 + 1] = _right_0;
        e.mem[_t_0 + 2] = 0;
        return term_tsk(FID_ir_valueMember, _t_0);
      }
      r0 = _left_0;
      r1 = _right_0;
      r2 = 0;
      WL_JMP(FID_ir_valueMember);
    } else if ((_op_0 & 3) == 1) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryValues)) {
        u64 _t_1 = task_node(e, FID_ir_setBinaryValues, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _op_0;
        e.mem[_t_1 + 1] = _left_0;
        e.mem[_t_1 + 2] = _right_0;
        return term_tsk(FID_ir_setBinaryValues, _t_1);
      }
      r0 = _op_0;
      r1 = _left_0;
      r2 = _right_0;
      WL_JMP(FID_ir_setBinaryValues);
    } else if (_op_0 == 7) {
      if (term_aux(_left_0) == CID_ir_VE) {
        Term _fb_0[1];
        u64 _sp_0 = ctr_take(e, _left_0, 1, _fb_0);
        Term _f_0 = _fb_0[0];
        if (term_aux(_right_0) == CID_ir_VE) {
          Term _fb_1[1];
          u64 _sp_1 = ctr_take(e, _right_0, 1, _fb_1);
          Term _f_1 = _fb_1[0];
          term_sink(e, _f_1);
          u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
          e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
          spare_free(e, cls_fit(1), _sp_1);
          r0 = term_ctr(CID_ir_VE, _nd_0);
          WL_RETN(1);
        } else {
          term_sink(e, _right_0);
          u64 _nd_1 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
          e.mem[_nd_1 + 0] = rfc_seal(e, _f_0);
          r0 = term_ctr(CID_ir_VE, _nd_1);
          WL_RETN(1);
        }
      } else if (term_aux(_left_0) == CID_ir_VI) {
        u32 _lo_0 = term_loc(_left_0);
        if (term_aux(_right_0) == CID_ir_VE) {
          Term _fb_2[1];
          u64 _sp_2 = ctr_take(e, _right_0, 1, _fb_2);
          Term _f_2 = _fb_2[0];
          u64 _nd_2 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 0);
          e.mem[_nd_2 + 0] = rfc_seal(e, _f_2);
          r0 = term_ctr(CID_ir_VE, _nd_2);
          WL_RETN(1);
        } else if (term_aux(_right_0) == CID_ir_VI) {
          u32 _hi_0 = term_loc(_right_0);
          Term _o_0[1];
          if (spin_ir_sle(e, _o_0, _lo_0, _hi_0) == 0) {
            return 0;
          }
          if ((u32)_o_0[0] == 0) {
            Term _o_1[1];
            if (spin_ir_emptySet(e, _o_1) == 0) {
              return 0;
            }
            r0 = _o_1[0];
            WL_RETN(1);
          } else {
            u32 _distance_0 = U32_BIN(_hi_0, -, _lo_0);
            u32 _s_0 = U32_BIN(_distance_0, <, 1048576ull);
            if (_s_0 == 0) {
              u32 _size_0 = U32_BIN(_distance_0, +, 1);
              r0 = term_ctr(CID_ir_VE, STAT_OFF + 821);
              WL_RETN(1);
            } else {
              u32 _size_1 = U32_BIN(_distance_0, +, 1);
              if (seq) {
                WL_ROOM(3);
                STK(0) = _lo_0;
                STK(1) = _size_1;
                STK(2) = FID_ir_setBinary_k1;
                WL_PUSHN(3);
              } else {
                u64 _t_2 = task_node(e, FID_ir_setBinary_k1, WL_CONT, WL_IDX, 1);
                e.mem[_t_2 + 0] = _lo_0;
                e.mem[_t_2 + 1] = _size_1;
                WL_CONT = term_tsk(FID_ir_setBinary_k1, _t_2);
                WL_IDX = 2;
              }
              if (!DEVICE && !seq && fid_nofk(FID_ir_rangeItems)) {
                u64 _t_3 = task_node(e, FID_ir_rangeItems, WL_CONT, WL_IDX, 0);
                e.mem[_t_3 + 0] = ((u64)(_size_1));
                e.mem[_t_3 + 1] = _lo_0;
                return term_tsk(FID_ir_rangeItems, _t_3);
              }
              r0 = ((u64)(_size_1));
              r1 = _lo_0;
              WL_JMP(FID_ir_rangeItems);
            }
          }
        } else {
          term_sink(e, _right_0);
          r0 = term_ctr(CID_ir_VE, STAT_OFF + 1526);
          WL_RETN(1);
        }
      } else {
        term_sink(e, _left_0);
        if (term_aux(_right_0) == CID_ir_VE) {
          Term _fb_3[1];
          u64 _sp_3 = ctr_take(e, _right_0, 1, _fb_3);
          Term _f_3 = _fb_3[0];
          u64 _nd_5 = _sp_3 >= HEAP_OFF ? _sp_3 : heap_alloc(e, 0);
          e.mem[_nd_5 + 0] = rfc_seal(e, _f_3);
          r0 = term_ctr(CID_ir_VE, _nd_5);
          WL_RETN(1);
        } else {
          term_sink(e, _right_0);
          r0 = term_ctr(CID_ir_VE, STAT_OFF + 1526);
          WL_RETN(1);
        }
      }
    } else if ((_op_0 & 3) == 3) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryValues)) {
        u64 _t_4 = task_node(e, FID_ir_setBinaryValues, WL_CONT, WL_IDX, 0);
        e.mem[_t_4 + 0] = _op_0;
        e.mem[_t_4 + 1] = _left_0;
        e.mem[_t_4 + 2] = _right_0;
        return term_tsk(FID_ir_setBinaryValues, _t_4);
      }
      r0 = _op_0;
      r1 = _left_0;
      r2 = _right_0;
      WL_JMP(FID_ir_setBinaryValues);
    } else if (_op_0 == 6) {
      if (!DEVICE && !seq && fid_nofk(FID_ir_valueMember)) {
        u64 _t_5 = task_node(e, FID_ir_valueMember, WL_CONT, WL_IDX, 0);
        e.mem[_t_5 + 0] = _left_0;
        e.mem[_t_5 + 1] = _right_0;
        e.mem[_t_5 + 2] = 1;
        return term_tsk(FID_ir_valueMember, _t_5);
      }
      r0 = _left_0;
      r1 = _right_0;
      r2 = 1;
      WL_JMP(FID_ir_valueMember);
    } else {
      if (!DEVICE && !seq && fid_nofk(FID_ir_setBinaryValues)) {
        u64 _t_6 = task_node(e, FID_ir_setBinaryValues, WL_CONT, WL_IDX, 0);
        e.mem[_t_6 + 0] = _op_0;
        e.mem[_t_6 + 1] = _left_0;
        e.mem[_t_6 + 2] = _right_0;
        return term_tsk(FID_ir_setBinaryValues, _t_6);
      }
      r0 = _op_0;
      r1 = _left_0;
      r2 = _right_0;
      WL_JMP(FID_ir_setBinaryValues);
    }
  }}

  WL_CASE(FID_ir_setBinary_k1)
    WL_POPN(2);
    u32 _lo_1 = STK(0);
    u32 _size_2 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_3 = heap_alloc(e, 1);
    e.mem[_nd_3 + 0] = rfc_seal(e, _lo_1);
    e.mem[_nd_3 + 1] = rfc_seal(e, _size_2);
    u64 _nd_4 = heap_alloc(e, 1);
    e.mem[_nd_4 + 0] = rfc_seal(e, _h_0);
    e.mem[_nd_4 + 1] = rfc_seal(e, term_ctr(CID_ir_IntDomain, _nd_3));
    r0 = term_ctr(CID_ir_VSet, _nd_4);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_evalSetItems)
    Term _nodes_0 = r0;
    Term _env_0 = r1;
    Term _env_1 = r2;
    Term _items_0 = r3;
    WL_OPEN
    if (term_aux(_items_0) == CID_Nil) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      r0 = term_pak(CID_Nil, 0);
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _items_0, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _f_1;
        STK(4) = FID_ir_evalSetItems_k1;
        WL_PUSHN(5);
      } else {
        u64 _t_0 = task_node(e, FID_ir_evalSetItems_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _nodes_0;
        e.mem[_t_0 + 1] = _env_0;
        e.mem[_t_0 + 2] = _env_1;
        e.mem[_t_0 + 3] = _f_1;
        WL_CONT = term_tsk(FID_ir_evalSetItems_k1, _t_0);
        WL_IDX = 4;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
        u64 _t_1 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _nodes_0;
        e.mem[_t_1 + 1] = _env_0;
        e.mem[_t_1 + 2] = _env_1;
        e.mem[_t_1 + 3] = _f_0;
        return term_tsk(FID_ir_eval, _t_1);
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _f_0;
      WL_JMP(FID_ir_eval);
    }
  }}

  WL_CASE(FID_ir_evalSetItems_k1)
    WL_POPN(4);
    Term _nodes_1 = STK(0);
    Term _env_2 = STK(1);
    Term _env_3 = STK(2);
    Term _f_2 = STK(3);
    Term _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_0;
      STK(1) = FID_ir_evalSetItems_k3;
      WL_PUSHN(2);
    } else {
      u64 _t_2 = task_node(e, FID_ir_evalSetItems_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _h_0;
      WL_CONT = term_tsk(FID_ir_evalSetItems_k3, _t_2);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_evalSetItems)) {
      u64 _t_3 = task_node(e, FID_ir_evalSetItems, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _nodes_1;
      e.mem[_t_3 + 1] = _env_2;
      e.mem[_t_3 + 2] = _env_3;
      e.mem[_t_3 + 3] = _f_2;
      return term_tsk(FID_ir_evalSetItems, _t_3);
    }
    r0 = _nodes_1;
    r1 = _env_2;
    r2 = _env_3;
    r3 = _f_2;
    WL_JMP(FID_ir_evalSetItems);
  }}

  WL_CASE(FID_ir_evalSetItems_k3)
    WL_POPN(1);
    Term _h_2 = STK(0);
    Term _h_1 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _h_2);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_1);
    r0 = term_ctr(CID_Con, _nd_0);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_makeSet)
    Term _values_0 = r0;
    WL_OPEN
    _values_0 = term_keep(e, _values_0, 1);
    Term _o_0[2];
    if (spin_ir_setInputErrorValue(e, _o_0, _values_0) == 0) {
      return 0;
    }
    if ((u32)_o_0[0] == 1) {
      term_sink(e, _values_0);
      u64 _nd_0 = heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _o_0[1]);
      r0 = term_ctr(CID_ir_VE, _nd_0);
      WL_RETN(1);
    } else {
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_makeSet_k1;
        WL_PUSHN(1);
      } else {
        u64 _t_0 = task_node(e, FID_ir_makeSet_k1, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_makeSet_k1, _t_0);
        WL_IDX = 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_List_sort_0)) {
        u64 _t_1 = task_node(e, FID_List_sort_0, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _values_0;
        return term_tsk(FID_List_sort_0, _t_1);
      }
      r0 = _values_0;
      WL_JMP(FID_List_sort_0);
    }
  }}

  WL_CASE(FID_ir_makeSet_k1)
    Term _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_setUnique)) {
      u64 _t_2 = task_node(e, FID_ir_setUnique, WL_CONT, WL_IDX, 0);
      e.mem[_t_2 + 0] = _h_0;
      e.mem[_t_2 + 1] = 0;
      e.mem[_t_2 + 2] = 0;
      e.mem[_t_2 + 3] = term_pak(CID_Nil, 0);
      e.mem[_t_2 + 4] = 0ull;
      return term_tsk(FID_ir_setUnique, _t_2);
    }
    r0 = _h_0;
    r1 = 0;
    r2 = 0;
    r3 = term_pak(CID_Nil, 0);
    r4 = 0ull;
    WL_JMP(FID_ir_setUnique);
  }}

  WL_CASE(FID_U32_show_go)
    Term _f_0 = r0;
    u32 _n_0 = r1;
    Term _acc_0 = r2;
    WL_OPEN
    WL_SPIN
    if (_f_0 == 0) {
      r0 = _acc_0;
      WL_RETN(1);
    } else {
      Term _g_0 = (_f_0 - 1);
      u32 _s_0 = U32_BIN(_n_0, ==, 0);
      if (_s_0 == 1) {
        r0 = _acc_0;
        WL_RETN(1);
      } else {
        Term _a_0 = ((u32)(10ull) == 0 ? _n_0 : U32_BIN(_n_0, -, U32_QUO((u32)(_n_0), (u32)(10ull)) * 10ull));
        u64 _nd_0 = heap_alloc(e, 1);
        e.mem[_nd_0 + 0] = rfc_seal(e, U32_BIN(48ull, +, _a_0));
        e.mem[_nd_0 + 1] = rfc_seal(e, _acc_0);
        r0 = _g_0;
        r1 = ((u32)(10ull) == 0 ? 0 : (u64)U32_QUO((u32)(_n_0), (u32)(10ull)));
        r2 = term_ctr(CID_SCon, _nd_0);
        _f_0 = r0;
        _n_0 = r1;
        _acc_0 = r2;
        WL_AGAIN(FID_U32_show_go);
      }
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_eval)
    Term _nodes_0 = r0;
    Term _env_0 = r1;
    Term _env_1 = r2;
    u32 _id_0 = r3;
    WL_OPEN
    WL_SPIN
    _nodes_0 = term_keep(e, _nodes_0, 1);
    Term _o_0[5];
    if (spin_ir_nthNode(e, _o_0, _nodes_0, ((u64)(_id_0))) == 0) {
      return 0;
    }
    if ((u32)_o_0[0] == 0) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      r0 = term_pak(CID_ir_VI, _o_0[1]);
      WL_RETN(1);
    } else if ((u32)_o_0[0] == 1) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      r0 = term_pak(CID_ir_VB, _o_0[1]);
      WL_RETN(1);
    } else if ((u32)_o_0[0] == 2) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      u64 _nd_0 = heap_alloc(e, 0);
      e.mem[_nd_0 + 0] = rfc_seal(e, _o_0[1]);
      r0 = term_ctr(CID_ir_VS, _nd_0);
      WL_RETN(1);
    } else if ((u32)_o_0[0] == 3) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_1);
      if (seq) {
        WL_ROOM(3);
        STK(0) = _env_0;
        STK(1) = _o_0[1];
        STK(2) = FID_ir_eval_k1;
        WL_PUSHN(3);
      } else {
        u64 _t_0 = task_node(e, FID_ir_eval_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _env_0;
        e.mem[_t_0 + 1] = _o_0[1];
        WL_CONT = term_tsk(FID_ir_eval_k1, _t_0);
        WL_IDX = 2;
      }
      if (!DEVICE && !seq && fid_nofk(FID_U32_show)) {
        u64 _t_1 = task_node(e, FID_U32_show, WL_CONT, WL_IDX, 0);
        e.mem[_t_1 + 0] = _o_0[1];
        return term_tsk(FID_U32_show, _t_1);
      }
      r0 = _o_0[1];
      WL_JMP(FID_U32_show);
    } else if ((u32)_o_0[0] == 4) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      r0 = _o_0[1];
      WL_RETN(1);
    } else if ((u32)_o_0[0] == 5) {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      if (!DEVICE && !seq && fid_nofk(FID_ir_localFind)) {
        u64 _t_6 = task_node(e, FID_ir_localFind, WL_CONT, WL_IDX, 0);
        e.mem[_t_6 + 0] = _env_1;
        e.mem[_t_6 + 1] = _o_0[1];
        return term_tsk(FID_ir_localFind, _t_6);
      }
      r0 = _env_1;
      r1 = _o_0[1];
      WL_JMP(FID_ir_localFind);
    } else if ((u32)_o_0[0] == 6) {
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_eval_k7;
        WL_PUSHN(1);
      } else {
        u64 _t_7 = task_node(e, FID_ir_eval_k7, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_eval_k7, _t_7);
        WL_IDX = 0;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_evalSetItems)) {
        u64 _t_8 = task_node(e, FID_ir_evalSetItems, WL_CONT, WL_IDX, 0);
        e.mem[_t_8 + 0] = _nodes_0;
        e.mem[_t_8 + 1] = _env_0;
        e.mem[_t_8 + 2] = _env_1;
        e.mem[_t_8 + 3] = _o_0[1];
        return term_tsk(FID_ir_evalSetItems, _t_8);
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      WL_JMP(FID_ir_evalSetItems);
    } else if ((u32)_o_0[0] == 7) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(6);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[1];
        STK(4) = (u32)_o_0[3];
        STK(5) = FID_ir_eval_k9;
        WL_PUSHN(6);
      } else {
        u64 _t_10 = task_node(e, FID_ir_eval_k9, WL_CONT, WL_IDX, 1);
        e.mem[_t_10 + 0] = _nodes_0;
        e.mem[_t_10 + 1] = _env_0;
        e.mem[_t_10 + 2] = _env_1;
        e.mem[_t_10 + 3] = _o_0[1];
        e.mem[_t_10 + 4] = (u32)_o_0[3];
        WL_CONT = term_tsk(FID_ir_eval_k9, _t_10);
        WL_IDX = 5;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[2];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 8) {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _o_0[1];
        STK(1) = FID_ir_eval_k13;
        WL_PUSHN(2);
      } else {
        u64 _t_14 = task_node(e, FID_ir_eval_k13, WL_CONT, WL_IDX, 1);
        e.mem[_t_14 + 0] = _o_0[1];
        WL_CONT = term_tsk(FID_ir_eval_k13, _t_14);
        WL_IDX = 1;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[2];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 9) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(7);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[1];
        STK(4) = _o_0[2];
        STK(5) = (u32)_o_0[4];
        STK(6) = FID_ir_eval_k17;
        WL_PUSHN(7);
      } else {
        u64 _t_19 = task_node(e, FID_ir_eval_k17, WL_CONT, WL_IDX, 1);
        e.mem[_t_19 + 0] = _nodes_0;
        e.mem[_t_19 + 1] = _env_0;
        e.mem[_t_19 + 2] = _env_1;
        e.mem[_t_19 + 3] = _o_0[1];
        e.mem[_t_19 + 4] = _o_0[2];
        e.mem[_t_19 + 5] = (u32)_o_0[4];
        WL_CONT = term_tsk(FID_ir_eval_k17, _t_19);
        WL_IDX = 6;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = (u32)_o_0[3];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 10) {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _o_0[1];
        STK(1) = FID_ir_eval_k19;
        WL_PUSHN(2);
      } else {
        u64 _t_21 = task_node(e, FID_ir_eval_k19, WL_CONT, WL_IDX, 1);
        e.mem[_t_21 + 0] = _o_0[1];
        WL_CONT = term_tsk(FID_ir_eval_k19, _t_21);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_evalSetItems)) {
        u64 _t_22 = task_node(e, FID_ir_evalSetItems, WL_CONT, WL_IDX, 0);
        e.mem[_t_22 + 0] = _nodes_0;
        e.mem[_t_22 + 1] = _env_0;
        e.mem[_t_22 + 2] = _env_1;
        e.mem[_t_22 + 3] = _o_0[2];
        return term_tsk(FID_ir_evalSetItems, _t_22);
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[2];
      WL_JMP(FID_ir_evalSetItems);
    } else if ((u32)_o_0[0] == 11) {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _o_0[1];
        STK(1) = FID_ir_eval_k21;
        WL_PUSHN(2);
      } else {
        u64 _t_24 = task_node(e, FID_ir_eval_k21, WL_CONT, WL_IDX, 1);
        e.mem[_t_24 + 0] = _o_0[1];
        WL_CONT = term_tsk(FID_ir_eval_k21, _t_24);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_evalSetItems)) {
        u64 _t_25 = task_node(e, FID_ir_evalSetItems, WL_CONT, WL_IDX, 0);
        e.mem[_t_25 + 0] = _nodes_0;
        e.mem[_t_25 + 1] = _env_0;
        e.mem[_t_25 + 2] = _env_1;
        e.mem[_t_25 + 3] = _o_0[2];
        return term_tsk(FID_ir_evalSetItems, _t_25);
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[2];
      WL_JMP(FID_ir_evalSetItems);
    } else if ((u32)_o_0[0] == 12) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k29;
        WL_PUSHN(5);
      } else {
        u64 _t_33 = task_node(e, FID_ir_eval_k29, WL_CONT, WL_IDX, 1);
        e.mem[_t_33 + 0] = _nodes_0;
        e.mem[_t_33 + 1] = _env_0;
        e.mem[_t_33 + 2] = _env_1;
        e.mem[_t_33 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k29, _t_33);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 13) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(7);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = (u32)_o_0[3];
        STK(5) = (u32)_o_0[4];
        STK(6) = FID_ir_eval_k35;
        WL_PUSHN(7);
      } else {
        u64 _t_38 = task_node(e, FID_ir_eval_k35, WL_CONT, WL_IDX, 1);
        e.mem[_t_38 + 0] = _nodes_0;
        e.mem[_t_38 + 1] = _env_0;
        e.mem[_t_38 + 2] = _env_1;
        e.mem[_t_38 + 3] = _o_0[2];
        e.mem[_t_38 + 4] = (u32)_o_0[3];
        e.mem[_t_38 + 5] = (u32)_o_0[4];
        WL_CONT = term_tsk(FID_ir_eval_k35, _t_38);
        WL_IDX = 6;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 14) {
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_eval_k45;
        WL_PUSHN(1);
      } else {
        u64 _t_47 = task_node(e, FID_ir_eval_k45, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_eval_k45, _t_47);
        WL_IDX = 0;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 15) {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _o_0[1];
        STK(1) = FID_ir_eval_k49;
        WL_PUSHN(2);
      } else {
        u64 _t_51 = task_node(e, FID_ir_eval_k49, WL_CONT, WL_IDX, 1);
        e.mem[_t_51 + 0] = _o_0[1];
        WL_CONT = term_tsk(FID_ir_eval_k49, _t_51);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_evalSetItems)) {
        u64 _t_52 = task_node(e, FID_ir_evalSetItems, WL_CONT, WL_IDX, 0);
        e.mem[_t_52 + 0] = _nodes_0;
        e.mem[_t_52 + 1] = _env_0;
        e.mem[_t_52 + 2] = _env_1;
        e.mem[_t_52 + 3] = _o_0[2];
        return term_tsk(FID_ir_evalSetItems, _t_52);
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[2];
      WL_JMP(FID_ir_evalSetItems);
    } else if ((u32)_o_0[0] == 16) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k67;
        WL_PUSHN(5);
      } else {
        u64 _t_72 = task_node(e, FID_ir_eval_k67, WL_CONT, WL_IDX, 1);
        e.mem[_t_72 + 0] = _nodes_0;
        e.mem[_t_72 + 1] = _env_0;
        e.mem[_t_72 + 2] = _env_1;
        e.mem[_t_72 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k67, _t_72);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 17) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k71;
        WL_PUSHN(5);
      } else {
        u64 _t_75 = task_node(e, FID_ir_eval_k71, WL_CONT, WL_IDX, 1);
        e.mem[_t_75 + 0] = _nodes_0;
        e.mem[_t_75 + 1] = _env_0;
        e.mem[_t_75 + 2] = _env_1;
        e.mem[_t_75 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k71, _t_75);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 18) {
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_eval_k75;
        WL_PUSHN(1);
      } else {
        u64 _t_78 = task_node(e, FID_ir_eval_k75, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_eval_k75, _t_78);
        WL_IDX = 0;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 19) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k77;
        WL_PUSHN(5);
      } else {
        u64 _t_79 = task_node(e, FID_ir_eval_k77, WL_CONT, WL_IDX, 1);
        e.mem[_t_79 + 0] = _nodes_0;
        e.mem[_t_79 + 1] = _env_0;
        e.mem[_t_79 + 2] = _env_1;
        e.mem[_t_79 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k77, _t_79);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 20) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k81;
        WL_PUSHN(5);
      } else {
        u64 _t_83 = task_node(e, FID_ir_eval_k81, WL_CONT, WL_IDX, 1);
        e.mem[_t_83 + 0] = _nodes_0;
        e.mem[_t_83 + 1] = _env_0;
        e.mem[_t_83 + 2] = _env_1;
        e.mem[_t_83 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k81, _t_83);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 21) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k87;
        WL_PUSHN(5);
      } else {
        u64 _t_88 = task_node(e, FID_ir_eval_k87, WL_CONT, WL_IDX, 1);
        e.mem[_t_88 + 0] = _nodes_0;
        e.mem[_t_88 + 1] = _env_0;
        e.mem[_t_88 + 2] = _env_1;
        e.mem[_t_88 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k87, _t_88);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 22) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k91;
        WL_PUSHN(5);
      } else {
        u64 _t_91 = task_node(e, FID_ir_eval_k91, WL_CONT, WL_IDX, 1);
        e.mem[_t_91 + 0] = _nodes_0;
        e.mem[_t_91 + 1] = _env_0;
        e.mem[_t_91 + 2] = _env_1;
        e.mem[_t_91 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k91, _t_91);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 23) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k95;
        WL_PUSHN(5);
      } else {
        u64 _t_94 = task_node(e, FID_ir_eval_k95, WL_CONT, WL_IDX, 1);
        e.mem[_t_94 + 0] = _nodes_0;
        e.mem[_t_94 + 1] = _env_0;
        e.mem[_t_94 + 2] = _env_1;
        e.mem[_t_94 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k95, _t_94);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 24) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k99;
        WL_PUSHN(5);
      } else {
        u64 _t_97 = task_node(e, FID_ir_eval_k99, WL_CONT, WL_IDX, 1);
        e.mem[_t_97 + 0] = _nodes_0;
        e.mem[_t_97 + 1] = _env_0;
        e.mem[_t_97 + 2] = _env_1;
        e.mem[_t_97 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k99, _t_97);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 25) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k103;
        WL_PUSHN(5);
      } else {
        u64 _t_100 = task_node(e, FID_ir_eval_k103, WL_CONT, WL_IDX, 1);
        e.mem[_t_100 + 0] = _nodes_0;
        e.mem[_t_100 + 1] = _env_0;
        e.mem[_t_100 + 2] = _env_1;
        e.mem[_t_100 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k103, _t_100);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 26) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k107;
        WL_PUSHN(5);
      } else {
        u64 _t_103 = task_node(e, FID_ir_eval_k107, WL_CONT, WL_IDX, 1);
        e.mem[_t_103 + 0] = _nodes_0;
        e.mem[_t_103 + 1] = _env_0;
        e.mem[_t_103 + 2] = _env_1;
        e.mem[_t_103 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k107, _t_103);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 27) {
      if (seq) {
        WL_ROOM(1);
        STK(0) = FID_ir_eval_k111;
        WL_PUSHN(1);
      } else {
        u64 _t_106 = task_node(e, FID_ir_eval_k111, WL_CONT, WL_IDX, 1);
        WL_CONT = term_tsk(FID_ir_eval_k111, _t_106);
        WL_IDX = 0;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 28) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(5);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = FID_ir_eval_k113;
        WL_PUSHN(5);
      } else {
        u64 _t_107 = task_node(e, FID_ir_eval_k113, WL_CONT, WL_IDX, 1);
        e.mem[_t_107 + 0] = _nodes_0;
        e.mem[_t_107 + 1] = _env_0;
        e.mem[_t_107 + 2] = _env_1;
        e.mem[_t_107 + 3] = _o_0[2];
        WL_CONT = term_tsk(FID_ir_eval_k113, _t_107);
        WL_IDX = 4;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else if ((u32)_o_0[0] == 29) {
      _nodes_0 = term_keep(e, _nodes_0, 1);
      _env_0 = term_keep(e, _env_0, 1);
      _env_1 = term_keep(e, _env_1, 1);
      if (seq) {
        WL_ROOM(6);
        STK(0) = _nodes_0;
        STK(1) = _env_0;
        STK(2) = _env_1;
        STK(3) = _o_0[2];
        STK(4) = (u32)_o_0[3];
        STK(5) = FID_ir_eval_k117;
        WL_PUSHN(6);
      } else {
        u64 _t_110 = task_node(e, FID_ir_eval_k117, WL_CONT, WL_IDX, 1);
        e.mem[_t_110 + 0] = _nodes_0;
        e.mem[_t_110 + 1] = _env_0;
        e.mem[_t_110 + 2] = _env_1;
        e.mem[_t_110 + 3] = _o_0[2];
        e.mem[_t_110 + 4] = (u32)_o_0[3];
        WL_CONT = term_tsk(FID_ir_eval_k117, _t_110);
        WL_IDX = 5;
      }
      r0 = _nodes_0;
      r1 = _env_0;
      r2 = _env_1;
      r3 = _o_0[1];
      _nodes_0 = r0;
      _env_0 = r1;
      _env_1 = r2;
      _id_0 = r3;
      WL_AGAIN(FID_ir_eval);
    } else {
      term_sink(e, _nodes_0);
      term_sink(e, _env_0);
      term_sink(e, _env_1);
      u64 _nd_27 = heap_alloc(e, 0);
      e.mem[_nd_27 + 0] = rfc_seal(e, _o_0[1]);
      r0 = term_ctr(CID_ir_VE, _nd_27);
      WL_RETN(1);
    }
    WL_SPUN
  }}

  WL_CASE(FID_ir_eval_k1)
    WL_POPN(2);
    Term _env_2 = STK(0);
    u32 _o_1 = STK(1);
    Term _h_0 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(3);
      STK(0) = _env_2;
      STK(1) = _o_1;
      STK(2) = FID_ir_eval_k3;
      WL_PUSHN(3);
    } else {
      u64 _t_2 = task_node(e, FID_ir_eval_k3, WL_CONT, WL_IDX, 1);
      e.mem[_t_2 + 0] = _env_2;
      e.mem[_t_2 + 1] = _o_1;
      WL_CONT = term_tsk(FID_ir_eval_k3, _t_2);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_3 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _h_0;
      e.mem[_t_3 + 1] = term_ctr(CID_SCon, STAT_OFF + 2399);
      return term_tsk(FID_String_append, _t_3);
    }
    r0 = _h_0;
    r1 = term_ctr(CID_SCon, STAT_OFF + 2399);
    WL_JMP(FID_String_append);
  }}

  WL_CASE(FID_ir_eval_k3)
    WL_POPN(2);
    Term _env_3 = STK(0);
    u32 _o_2 = STK(1);
    Term _h_1 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(3);
      STK(0) = _env_3;
      STK(1) = _o_2;
      STK(2) = FID_ir_eval_k5;
      WL_PUSHN(3);
    } else {
      u64 _t_4 = task_node(e, FID_ir_eval_k5, WL_CONT, WL_IDX, 1);
      e.mem[_t_4 + 0] = _env_3;
      e.mem[_t_4 + 1] = _o_2;
      WL_CONT = term_tsk(FID_ir_eval_k5, _t_4);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_5 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_5 + 0] = term_ctr(CID_SCon, STAT_OFF + 2421);
      e.mem[_t_5 + 1] = _h_1;
      return term_tsk(FID_String_append, _t_5);
    }
    r0 = term_ctr(CID_SCon, STAT_OFF + 2421);
    r1 = _h_1;
    WL_JMP(FID_String_append);
  }}

  WL_CASE(FID_ir_eval_k5)
    WL_POPN(2);
    Term _env_4 = STK(0);
    u32 _o_3 = STK(1);
    Term _h_2 = r0;
    WL_OPEN
    u64 _nd_1 = heap_alloc(e, 0);
    e.mem[_nd_1 + 0] = rfc_seal(e, _h_2);
    Term _o_4[1];
    if (spin_ir_nthVal(e, _o_4, _env_4, ((u64)(_o_3)), term_ctr(CID_ir_VE, _nd_1)) == 0) {
      return 0;
    }
    r0 = _o_4[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k7)
    Term _h_3 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_makeSet)) {
      u64 _t_9 = task_node(e, FID_ir_makeSet, WL_CONT, WL_IDX, 0);
      e.mem[_t_9 + 0] = _h_3;
      return term_tsk(FID_ir_makeSet, _t_9);
    }
    r0 = _h_3;
    WL_JMP(FID_ir_makeSet);
  }}

  WL_CASE(FID_ir_eval_k9)
    WL_POPN(5);
    Term _nodes_1 = STK(0);
    Term _env_5 = STK(1);
    Term _env_6 = STK(2);
    u32 _o_5 = STK(3);
    u32 _o_6 = STK(4);
    Term _h_4 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(3);
      STK(0) = _o_5;
      STK(1) = _h_4;
      STK(2) = FID_ir_eval_k11;
      WL_PUSHN(3);
    } else {
      u64 _t_11 = task_node(e, FID_ir_eval_k11, WL_CONT, WL_IDX, 1);
      e.mem[_t_11 + 0] = _o_5;
      e.mem[_t_11 + 1] = _h_4;
      WL_CONT = term_tsk(FID_ir_eval_k11, _t_11);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_12 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_12 + 0] = _nodes_1;
      e.mem[_t_12 + 1] = _env_5;
      e.mem[_t_12 + 2] = _env_6;
      e.mem[_t_12 + 3] = _o_6;
      return term_tsk(FID_ir_eval, _t_12);
    }
    r0 = _nodes_1;
    r1 = _env_5;
    r2 = _env_6;
    r3 = _o_6;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k11)
    WL_POPN(2);
    u32 _o_7 = STK(0);
    Term _h_6 = STK(1);
    Term _h_5 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_setBinary)) {
      u64 _t_13 = task_node(e, FID_ir_setBinary, WL_CONT, WL_IDX, 0);
      e.mem[_t_13 + 0] = _o_7;
      e.mem[_t_13 + 1] = _h_6;
      e.mem[_t_13 + 2] = _h_5;
      return term_tsk(FID_ir_setBinary, _t_13);
    }
    r0 = _o_7;
    r1 = _h_6;
    r2 = _h_5;
    WL_JMP(FID_ir_setBinary);
  }}

  WL_CASE(FID_ir_eval_k13)
    WL_POPN(1);
    u32 _o_8 = STK(0);
    Term _h_7 = r0;
    WL_OPEN
    if (term_aux(_h_7) == CID_ir_VE) {
      Term _fb_0[1];
      u64 _sp_0 = ctr_take(e, _h_7, 1, _fb_0);
      Term _f_0 = _fb_0[0];
      u64 _nd_2 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 0);
      e.mem[_nd_2 + 0] = rfc_seal(e, _f_0);
      r0 = term_ctr(CID_ir_VE, _nd_2);
      WL_RETN(1);
    } else if (term_aux(_h_7) == CID_ir_VSet) {
      Term _fb_1[2];
      u64 _sp_1 = ctr_take(e, _h_7, 2, _fb_1);
      Term _f_1 = _fb_1[0];
      Term _f_2 = _fb_1[1];
      if (_o_8 == 1) {
        term_sink(e, _f_1);
        Term _o_9[1];
        if (spin_ir_domainSize(e, _o_9, _f_2) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(2), _sp_1);
        r0 = term_pak(CID_ir_VI, (u32)_o_9[0]);
        WL_RETN(1);
      } else if ((_o_8 & 3) == 1) {
        term_sink(e, _f_1);
        term_sink(e, _f_2);
        spare_free(e, cls_fit(2), _sp_1);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1447);
        WL_RETN(1);
      } else if (_o_8 == 3) {
        term_sink(e, _f_2);
        spare_free(e, cls_fit(2), _sp_1);
        if (!DEVICE && !seq && fid_nofk(FID_ir_unionItems)) {
          u64 _t_15 = task_node(e, FID_ir_unionItems, WL_CONT, WL_IDX, 0);
          e.mem[_t_15 + 0] = _f_1;
          e.mem[_t_15 + 1] = term_ctr(CID_ir_VSet, STAT_OFF + 795);
          return term_tsk(FID_ir_unionItems, _t_15);
        }
        r0 = _f_1;
        r1 = term_ctr(CID_ir_VSet, STAT_OFF + 795);
        WL_JMP(FID_ir_unionItems);
      } else if ((_o_8 & 3) == 3) {
        term_sink(e, _f_1);
        term_sink(e, _f_2);
        spare_free(e, cls_fit(2), _sp_1);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1447);
        WL_RETN(1);
      } else if (_o_8 == 2) {
        Term _o_10[1];
        if (spin_ir_domainSize(e, _o_10, _f_2) == 0) {
          return 0;
        }
        spare_free(e, cls_fit(2), _sp_1);
        if (seq) {
          WL_ROOM(2);
          STK(0) = _f_1;
          STK(1) = FID_ir_eval_k15;
          WL_PUSHN(2);
        } else {
          u64 _t_16 = task_node(e, FID_ir_eval_k15, WL_CONT, WL_IDX, 1);
          e.mem[_t_16 + 0] = _f_1;
          WL_CONT = term_tsk(FID_ir_eval_k15, _t_16);
          WL_IDX = 1;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_powerCount)) {
          u64 _t_17 = task_node(e, FID_ir_powerCount, WL_CONT, WL_IDX, 0);
          e.mem[_t_17 + 0] = (u32)_o_10[0];
          e.mem[_t_17 + 1] = 1ull;
          return term_tsk(FID_ir_powerCount, _t_17);
        }
        r0 = (u32)_o_10[0];
        r1 = 1ull;
        WL_JMP(FID_ir_powerCount);
      } else if ((_o_8 & 3) == 2) {
        term_sink(e, _f_1);
        term_sink(e, _f_2);
        spare_free(e, cls_fit(2), _sp_1);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1447);
        WL_RETN(1);
      } else if (_o_8 == 4) {
        term_sink(e, _f_1);
        term_sink(e, _f_2);
        spare_free(e, cls_fit(2), _sp_1);
        r0 = term_pak(CID_ir_VB, 1);
        WL_RETN(1);
      } else {
        term_sink(e, _f_1);
        term_sink(e, _f_2);
        spare_free(e, cls_fit(2), _sp_1);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 1447);
        WL_RETN(1);
      }
    } else {
      term_sink(e, _h_7);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 2374);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_eval_k15)
    WL_POPN(1);
    Term _f_3 = STK(0);
    u32 _h_8 = r0;
    u32 _h_9 = r1;
    WL_OPEN
    if (_h_8 == 0) {
      term_sink(e, _f_3);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 1037);
      WL_RETN(1);
    } else {
      if (!DEVICE && !seq && fid_nofk(FID_ir_powerItems)) {
        u64 _t_18 = task_node(e, FID_ir_powerItems, WL_CONT, WL_IDX, 0);
        e.mem[_t_18 + 0] = _f_3;
        return term_tsk(FID_ir_powerItems, _t_18);
      }
      r0 = _f_3;
      WL_JMP(FID_ir_powerItems);
    }
  }}

  WL_CASE(FID_ir_eval_k17)
    WL_POPN(6);
    Term _nodes_2 = STK(0);
    Term _env_7 = STK(1);
    Term _env_8 = STK(2);
    u32 _o_11 = STK(3);
    u32 _o_12 = STK(4);
    u32 _o_13 = STK(5);
    Term _h_10 = r0;
    WL_OPEN
    if (term_aux(_h_10) == CID_ir_VE) {
      term_sink(e, _nodes_2);
      term_sink(e, _env_7);
      term_sink(e, _env_8);
      Term _fb_2[1];
      u64 _sp_2 = ctr_take(e, _h_10, 1, _fb_2);
      Term _f_4 = _fb_2[0];
      u64 _nd_3 = _sp_2 >= HEAP_OFF ? _sp_2 : heap_alloc(e, 0);
      e.mem[_nd_3 + 0] = rfc_seal(e, _f_4);
      r0 = term_ctr(CID_ir_VE, _nd_3);
      WL_RETN(1);
    } else if (term_aux(_h_10) == CID_ir_VSet) {
      Term _fb_3[2];
      u64 _sp_3 = ctr_take(e, _h_10, 2, _fb_3);
      Term _f_5 = _fb_3[0];
      Term _f_6 = _fb_3[1];
      term_sink(e, _f_6);
      spare_free(e, cls_fit(2), _sp_3);
      if (!DEVICE && !seq && fid_nofk(FID_ir_boundedLoop)) {
        u64 _t_20 = task_node(e, FID_ir_boundedLoop, WL_CONT, WL_IDX, 0);
        e.mem[_t_20 + 0] = _nodes_2;
        e.mem[_t_20 + 1] = _env_7;
        e.mem[_t_20 + 2] = _env_8;
        e.mem[_t_20 + 3] = _o_11;
        e.mem[_t_20 + 4] = _o_12;
        e.mem[_t_20 + 5] = _o_13;
        e.mem[_t_20 + 6] = _f_5;
        e.mem[_t_20 + 7] = term_pak(CID_Nil, 0);
        return term_tsk(FID_ir_boundedLoop, _t_20);
      }
      r0 = _nodes_2;
      r1 = _env_7;
      r2 = _env_8;
      r3 = _o_11;
      r4 = _o_12;
      r5 = _o_13;
      r6 = _f_5;
      r7 = term_pak(CID_Nil, 0);
      WL_JMP(FID_ir_boundedLoop);
    } else {
      term_sink(e, _nodes_2);
      term_sink(e, _env_7);
      term_sink(e, _env_8);
      term_sink(e, _h_10);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 2321);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_eval_k19)
    WL_POPN(1);
    Term _o_14 = STK(0);
    Term _h_11 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_functionSorted)) {
      u64 _t_23 = task_node(e, FID_ir_functionSorted, WL_CONT, WL_IDX, 0);
      e.mem[_t_23 + 0] = _o_14;
      e.mem[_t_23 + 1] = _h_11;
      return term_tsk(FID_ir_functionSorted, _t_23);
    }
    r0 = _o_14;
    r1 = _h_11;
    WL_JMP(FID_ir_functionSorted);
  }}

  WL_CASE(FID_ir_eval_k21)
    WL_POPN(1);
    Term _o_15 = STK(0);
    Term _h_12 = r0;
    WL_OPEN
    _h_12 = term_keep(e, _h_12, 1);
    if (seq) {
      WL_ROOM(3);
      STK(0) = _o_15;
      STK(1) = _h_12;
      STK(2) = FID_ir_eval_k23;
      WL_PUSHN(3);
    } else {
      u64 _t_26 = task_node(e, FID_ir_eval_k23, WL_CONT, WL_IDX, 1);
      e.mem[_t_26 + 0] = _o_15;
      e.mem[_t_26 + 1] = _h_12;
      WL_CONT = term_tsk(FID_ir_eval_k23, _t_26);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_productCount)) {
      u64 _t_27 = task_node(e, FID_ir_productCount, WL_CONT, WL_IDX, 0);
      e.mem[_t_27 + 0] = _h_12;
      e.mem[_t_27 + 1] = 1ull;
      return term_tsk(FID_ir_productCount, _t_27);
    }
    r0 = _h_12;
    r1 = 1ull;
    WL_JMP(FID_ir_productCount);
  }}

  WL_CASE(FID_ir_eval_k23)
    WL_POPN(2);
    Term _o_16 = STK(0);
    Term _h_14 = STK(1);
    Term _h_13 = r0;
    WL_OPEN
    if (term_aux(_h_13) == CID_ir_VE) {
      Term _fb_4[1];
      u64 _sp_4 = ctr_take(e, _h_13, 1, _fb_4);
      Term _f_7 = _fb_4[0];
      term_sink(e, _o_16);
      term_sink(e, _h_14);
      u64 _nd_4 = _sp_4 >= HEAP_OFF ? _sp_4 : heap_alloc(e, 0);
      e.mem[_nd_4 + 0] = rfc_seal(e, _f_7);
      r0 = term_ctr(CID_ir_VE, _nd_4);
      WL_RETN(1);
    } else if (term_aux(_h_13) == CID_ir_VI) {
      u32 _size_0 = term_loc(_h_13);
      u32 _s_0 = U32_BIN(_size_0, ==, 0ull);
      if (_s_0 == 1) {
        term_sink(e, _o_16);
        term_sink(e, _h_14);
        Term _o_17[1];
        if (spin_ir_emptySet(e, _o_17) == 0) {
          return 0;
        }
        r0 = _o_17[0];
        WL_RETN(1);
      } else {
        if (seq) {
          WL_ROOM(3);
          STK(0) = _size_0;
          STK(1) = _o_16;
          STK(2) = FID_ir_eval_k25;
          WL_PUSHN(3);
        } else {
          u64 _t_28 = task_node(e, FID_ir_eval_k25, WL_CONT, WL_IDX, 1);
          e.mem[_t_28 + 0] = _size_0;
          e.mem[_t_28 + 1] = _o_16;
          WL_CONT = term_tsk(FID_ir_eval_k25, _t_28);
          WL_IDX = 2;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_productRows)) {
          u64 _t_29 = task_node(e, FID_ir_productRows, WL_CONT, WL_IDX, 0);
          e.mem[_t_29 + 0] = _h_14;
          return term_tsk(FID_ir_productRows, _t_29);
        }
        r0 = _h_14;
        WL_JMP(FID_ir_productRows);
      }
    } else {
      term_sink(e, _h_13);
      term_sink(e, _o_16);
      term_sink(e, _h_14);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 1404);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_eval_k25)
    WL_POPN(2);
    u32 _size_1 = STK(0);
    Term _o_18 = STK(1);
    Term _h_15 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _size_1;
      STK(1) = FID_ir_eval_k27;
      WL_PUSHN(2);
    } else {
      u64 _t_30 = task_node(e, FID_ir_eval_k27, WL_CONT, WL_IDX, 1);
      e.mem[_t_30 + 0] = _size_1;
      WL_CONT = term_tsk(FID_ir_eval_k27, _t_30);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_productFunctions)) {
      u64 _t_31 = task_node(e, FID_ir_productFunctions, WL_CONT, WL_IDX, 0);
      e.mem[_t_31 + 0] = _o_18;
      e.mem[_t_31 + 1] = _h_15;
      return term_tsk(FID_ir_productFunctions, _t_31);
    }
    r0 = _o_18;
    r1 = _h_15;
    WL_JMP(FID_ir_productFunctions);
  }}

  WL_CASE(FID_ir_eval_k27)
    WL_POPN(1);
    u32 _size_2 = STK(0);
    Term _h_16 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
      u64 _t_32 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
      e.mem[_t_32 + 0] = _h_16;
      e.mem[_t_32 + 1] = _size_2;
      return term_tsk(FID_ir_setSorted, _t_32);
    }
    r0 = _h_16;
    r1 = _size_2;
    WL_JMP(FID_ir_setSorted);
  }}

  WL_CASE(FID_ir_eval_k29)
    WL_POPN(4);
    Term _nodes_3 = STK(0);
    Term _env_9 = STK(1);
    Term _env_10 = STK(2);
    u32 _o_19 = STK(3);
    Term _h_17 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_17;
      STK(1) = FID_ir_eval_k31;
      WL_PUSHN(2);
    } else {
      u64 _t_34 = task_node(e, FID_ir_eval_k31, WL_CONT, WL_IDX, 1);
      e.mem[_t_34 + 0] = _h_17;
      WL_CONT = term_tsk(FID_ir_eval_k31, _t_34);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_35 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_35 + 0] = _nodes_3;
      e.mem[_t_35 + 1] = _env_9;
      e.mem[_t_35 + 2] = _env_10;
      e.mem[_t_35 + 3] = _o_19;
      return term_tsk(FID_ir_eval, _t_35);
    }
    r0 = _nodes_3;
    r1 = _env_9;
    r2 = _env_10;
    r3 = _o_19;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k31)
    WL_POPN(1);
    Term _h_19 = STK(0);
    Term _h_18 = r0;
    WL_OPEN
    if (term_aux(_h_19) == CID_ir_VE) {
      Term _fb_5[1];
      u64 _sp_5 = ctr_take(e, _h_19, 1, _fb_5);
      Term _f_8 = _fb_5[0];
      if (term_aux(_h_18) == CID_ir_VE) {
        Term _fb_6[1];
        u64 _sp_6 = ctr_take(e, _h_18, 1, _fb_6);
        Term _f_9 = _fb_6[0];
        term_sink(e, _f_9);
        u64 _nd_5 = _sp_5 >= HEAP_OFF ? _sp_5 : heap_alloc(e, 0);
        e.mem[_nd_5 + 0] = rfc_seal(e, _f_8);
        spare_free(e, cls_fit(1), _sp_6);
        r0 = term_ctr(CID_ir_VE, _nd_5);
        WL_RETN(1);
      } else {
        term_sink(e, _h_18);
        u64 _nd_6 = _sp_5 >= HEAP_OFF ? _sp_5 : heap_alloc(e, 0);
        e.mem[_nd_6 + 0] = rfc_seal(e, _f_8);
        r0 = term_ctr(CID_ir_VE, _nd_6);
        WL_RETN(1);
      }
    } else if (term_aux(_h_19) == CID_ir_VFun) {
      Term _fb_7[2];
      u64 _sp_7 = ctr_take(e, _h_19, 2, _fb_7);
      Term _f_10 = _fb_7[0];
      u32 _f_11 = _fb_7[1];
      if (term_aux(_h_18) == CID_ir_VE) {
        term_sink(e, _f_10);
        Term _fb_8[1];
        u64 _sp_8 = ctr_take(e, _h_18, 1, _fb_8);
        Term _f_12 = _fb_8[0];
        u64 _nd_7 = _sp_8 >= HEAP_OFF ? _sp_8 : heap_alloc(e, 0);
        e.mem[_nd_7 + 0] = rfc_seal(e, _f_12);
        spare_free(e, cls_fit(2), _sp_7);
        r0 = term_ctr(CID_ir_VE, _nd_7);
        WL_RETN(1);
      } else {
        spare_free(e, cls_fit(2), _sp_7);
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_eval_k33;
          WL_PUSHN(1);
        } else {
          u64 _t_36 = task_node(e, FID_ir_eval_k33, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_eval_k33, _t_36);
          WL_IDX = 0;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_functionFind)) {
          u64 _t_37 = task_node(e, FID_ir_functionFind, WL_CONT, WL_IDX, 0);
          e.mem[_t_37 + 0] = _f_10;
          e.mem[_t_37 + 1] = _h_18;
          return term_tsk(FID_ir_functionFind, _t_37);
        }
        r0 = _f_10;
        r1 = _h_18;
        WL_JMP(FID_ir_functionFind);
      }
    } else {
      term_sink(e, _h_19);
      if (term_aux(_h_18) == CID_ir_VE) {
        Term _fb_9[1];
        u64 _sp_9 = ctr_take(e, _h_18, 1, _fb_9);
        Term _f_13 = _fb_9[0];
        u64 _nd_8 = _sp_9 >= HEAP_OFF ? _sp_9 : heap_alloc(e, 0);
        e.mem[_nd_8 + 0] = rfc_seal(e, _f_13);
        r0 = term_ctr(CID_ir_VE, _nd_8);
        WL_RETN(1);
      } else {
        term_sink(e, _h_18);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 2284);
        WL_RETN(1);
      }
    }
  }}

  WL_CASE(FID_ir_eval_k33)
    u32 _h_20 = r0;
    Term _h_21 = r1;
    WL_OPEN
    Term _o_20[1];
    if (spin_ir_functionFound(e, _o_20, _h_20, _h_21) == 0) {
      return 0;
    }
    r0 = _o_20[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k35)
    WL_POPN(6);
    Term _nodes_4 = STK(0);
    Term _env_11 = STK(1);
    Term _env_12 = STK(2);
    u32 _o_21 = STK(3);
    u32 _o_22 = STK(4);
    u32 _o_23 = STK(5);
    Term _h_22 = r0;
    WL_OPEN
    _nodes_4 = term_keep(e, _nodes_4, 1);
    _env_11 = term_keep(e, _env_11, 1);
    _env_12 = term_keep(e, _env_12, 1);
    if (seq) {
      WL_ROOM(7);
      STK(0) = _nodes_4;
      STK(1) = _env_11;
      STK(2) = _env_12;
      STK(3) = _o_22;
      STK(4) = _o_23;
      STK(5) = _h_22;
      STK(6) = FID_ir_eval_k37;
      WL_PUSHN(7);
    } else {
      u64 _t_39 = task_node(e, FID_ir_eval_k37, WL_CONT, WL_IDX, 1);
      e.mem[_t_39 + 0] = _nodes_4;
      e.mem[_t_39 + 1] = _env_11;
      e.mem[_t_39 + 2] = _env_12;
      e.mem[_t_39 + 3] = _o_22;
      e.mem[_t_39 + 4] = _o_23;
      e.mem[_t_39 + 5] = _h_22;
      WL_CONT = term_tsk(FID_ir_eval_k37, _t_39);
      WL_IDX = 6;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_40 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_40 + 0] = _nodes_4;
      e.mem[_t_40 + 1] = _env_11;
      e.mem[_t_40 + 2] = _env_12;
      e.mem[_t_40 + 3] = _o_21;
      return term_tsk(FID_ir_eval, _t_40);
    }
    r0 = _nodes_4;
    r1 = _env_11;
    r2 = _env_12;
    r3 = _o_21;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k37)
    WL_POPN(6);
    Term _nodes_5 = STK(0);
    Term _env_13 = STK(1);
    Term _env_14 = STK(2);
    u32 _o_24 = STK(3);
    u32 _o_25 = STK(4);
    Term _h_24 = STK(5);
    Term _h_23 = r0;
    WL_OPEN
    if (term_aux(_h_24) == CID_ir_VE) {
      term_sink(e, _nodes_5);
      term_sink(e, _env_13);
      term_sink(e, _env_14);
      Term _fb_10[1];
      u64 _sp_10 = ctr_take(e, _h_24, 1, _fb_10);
      Term _f_14 = _fb_10[0];
      if (term_aux(_h_23) == CID_ir_VE) {
        Term _fb_11[1];
        u64 _sp_11 = ctr_take(e, _h_23, 1, _fb_11);
        Term _f_15 = _fb_11[0];
        term_sink(e, _f_15);
        u64 _nd_9 = _sp_10 >= HEAP_OFF ? _sp_10 : heap_alloc(e, 0);
        e.mem[_nd_9 + 0] = rfc_seal(e, _f_14);
        spare_free(e, cls_fit(1), _sp_11);
        r0 = term_ctr(CID_ir_VE, _nd_9);
        WL_RETN(1);
      } else {
        term_sink(e, _h_23);
        u64 _nd_10 = _sp_10 >= HEAP_OFF ? _sp_10 : heap_alloc(e, 0);
        e.mem[_nd_10 + 0] = rfc_seal(e, _f_14);
        r0 = term_ctr(CID_ir_VE, _nd_10);
        WL_RETN(1);
      }
    } else if (term_aux(_h_24) == CID_ir_VFun) {
      Term _fb_12[2];
      u64 _sp_12 = ctr_take(e, _h_24, 2, _fb_12);
      Term _f_16 = _fb_12[0];
      u32 _f_17 = _fb_12[1];
      if (term_aux(_h_23) == CID_ir_VE) {
        term_sink(e, _nodes_5);
        term_sink(e, _env_13);
        term_sink(e, _env_14);
        term_sink(e, _f_16);
        Term _fb_13[1];
        u64 _sp_13 = ctr_take(e, _h_23, 1, _fb_13);
        Term _f_18 = _fb_13[0];
        u64 _nd_11 = _sp_13 >= HEAP_OFF ? _sp_13 : heap_alloc(e, 0);
        e.mem[_nd_11 + 0] = rfc_seal(e, _f_18);
        spare_free(e, cls_fit(2), _sp_12);
        r0 = term_ctr(CID_ir_VE, _nd_11);
        WL_RETN(1);
      } else {
        _f_16 = term_keep(e, _f_16, 1);
        _h_23 = term_keep(e, _h_23, 1);
        spare_free(e, cls_fit(2), _sp_12);
        if (seq) {
          WL_ROOM(9);
          STK(0) = _nodes_5;
          STK(1) = _env_13;
          STK(2) = _env_14;
          STK(3) = _o_24;
          STK(4) = _o_25;
          STK(5) = _f_16;
          STK(6) = _f_17;
          STK(7) = _h_23;
          STK(8) = FID_ir_eval_k39;
          WL_PUSHN(9);
        } else {
          u64 _t_41 = task_node(e, FID_ir_eval_k39, WL_CONT, WL_IDX, 1);
          e.mem[_t_41 + 0] = _nodes_5;
          e.mem[_t_41 + 1] = _env_13;
          e.mem[_t_41 + 2] = _env_14;
          e.mem[_t_41 + 3] = _o_24;
          e.mem[_t_41 + 4] = _o_25;
          e.mem[_t_41 + 5] = _f_16;
          e.mem[_t_41 + 6] = _f_17;
          e.mem[_t_41 + 7] = _h_23;
          WL_CONT = term_tsk(FID_ir_eval_k39, _t_41);
          WL_IDX = 8;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_functionFind)) {
          u64 _t_42 = task_node(e, FID_ir_functionFind, WL_CONT, WL_IDX, 0);
          e.mem[_t_42 + 0] = _f_16;
          e.mem[_t_42 + 1] = _h_23;
          return term_tsk(FID_ir_functionFind, _t_42);
        }
        r0 = _f_16;
        r1 = _h_23;
        WL_JMP(FID_ir_functionFind);
      }
    } else {
      term_sink(e, _nodes_5);
      term_sink(e, _env_13);
      term_sink(e, _env_14);
      term_sink(e, _h_24);
      if (term_aux(_h_23) == CID_ir_VE) {
        Term _fb_15[1];
        u64 _sp_15 = ctr_take(e, _h_23, 1, _fb_15);
        Term _f_25 = _fb_15[0];
        u64 _nd_15 = _sp_15 >= HEAP_OFF ? _sp_15 : heap_alloc(e, 0);
        e.mem[_nd_15 + 0] = rfc_seal(e, _f_25);
        r0 = term_ctr(CID_ir_VE, _nd_15);
        WL_RETN(1);
      } else {
        term_sink(e, _h_23);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 2271);
        WL_RETN(1);
      }
    }
  }}

  WL_CASE(FID_ir_eval_k39)
    WL_POPN(8);
    Term _nodes_6 = STK(0);
    Term _env_15 = STK(1);
    Term _env_16 = STK(2);
    u32 _o_26 = STK(3);
    u32 _o_27 = STK(4);
    Term _f_19 = STK(5);
    u32 _f_20 = STK(6);
    Term _h_27 = STK(7);
    u32 _h_25 = r0;
    Term _h_26 = r1;
    WL_OPEN
    if (_h_25 == 0) {
      term_sink(e, _nodes_6);
      term_sink(e, _env_15);
      term_sink(e, _env_16);
      term_sink(e, _h_27);
      u64 _nd_12 = heap_alloc(e, 1);
      e.mem[_nd_12 + 0] = rfc_seal(e, _f_19);
      e.mem[_nd_12 + 1] = rfc_seal(e, _f_20);
      r0 = term_ctr(CID_ir_VFun, _nd_12);
      WL_RETN(1);
    } else {
      Term _o_28[2];
      if (spin_ir_bindLocal(e, _o_28, _env_15, _env_16, _o_26, _h_26) == 0) {
        return 0;
      }
      if (seq) {
        WL_ROOM(4);
        STK(0) = _f_19;
        STK(1) = _f_20;
        STK(2) = _h_27;
        STK(3) = FID_ir_eval_k41;
        WL_PUSHN(4);
      } else {
        u64 _t_43 = task_node(e, FID_ir_eval_k41, WL_CONT, WL_IDX, 1);
        e.mem[_t_43 + 0] = _f_19;
        e.mem[_t_43 + 1] = _f_20;
        e.mem[_t_43 + 2] = _h_27;
        WL_CONT = term_tsk(FID_ir_eval_k41, _t_43);
        WL_IDX = 3;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
        u64 _t_44 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
        e.mem[_t_44 + 0] = _nodes_6;
        e.mem[_t_44 + 1] = _o_28[0];
        e.mem[_t_44 + 2] = _o_28[1];
        e.mem[_t_44 + 3] = _o_27;
        return term_tsk(FID_ir_eval, _t_44);
      }
      r0 = _nodes_6;
      r1 = _o_28[0];
      r2 = _o_28[1];
      r3 = _o_27;
      WL_JMP(FID_ir_eval);
    }
  }}

  WL_CASE(FID_ir_eval_k41)
    WL_POPN(3);
    Term _f_21 = STK(0);
    u32 _f_22 = STK(1);
    Term _h_29 = STK(2);
    Term _h_28 = r0;
    WL_OPEN
    if (term_aux(_h_28) == CID_ir_VE) {
      Term _fb_14[1];
      u64 _sp_14 = ctr_take(e, _h_28, 1, _fb_14);
      Term _f_23 = _fb_14[0];
      term_sink(e, _f_21);
      term_sink(e, _h_29);
      u64 _nd_13 = _sp_14 >= HEAP_OFF ? _sp_14 : heap_alloc(e, 0);
      e.mem[_nd_13 + 0] = rfc_seal(e, _f_23);
      r0 = term_ctr(CID_ir_VE, _nd_13);
      WL_RETN(1);
    } else {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_22;
        STK(1) = FID_ir_eval_k43;
        WL_PUSHN(2);
      } else {
        u64 _t_45 = task_node(e, FID_ir_eval_k43, WL_CONT, WL_IDX, 1);
        e.mem[_t_45 + 0] = _f_22;
        WL_CONT = term_tsk(FID_ir_eval_k43, _t_45);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionReplace)) {
        u64 _t_46 = task_node(e, FID_ir_functionReplace, WL_CONT, WL_IDX, 0);
        e.mem[_t_46 + 0] = _f_21;
        e.mem[_t_46 + 1] = _h_29;
        e.mem[_t_46 + 2] = _h_28;
        return term_tsk(FID_ir_functionReplace, _t_46);
      }
      r0 = _f_21;
      r1 = _h_29;
      r2 = _h_28;
      WL_JMP(FID_ir_functionReplace);
    }
  }}

  WL_CASE(FID_ir_eval_k43)
    WL_POPN(1);
    u32 _f_24 = STK(0);
    Term _h_30 = r0;
    WL_OPEN
    u64 _nd_14 = heap_alloc(e, 1);
    e.mem[_nd_14 + 0] = rfc_seal(e, _h_30);
    e.mem[_nd_14 + 1] = rfc_seal(e, _f_24);
    r0 = term_ctr(CID_ir_VFun, _nd_14);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k45)
    Term _h_31 = r0;
    WL_OPEN
    if (term_aux(_h_31) == CID_ir_VE) {
      Term _fb_16[1];
      u64 _sp_16 = ctr_take(e, _h_31, 1, _fb_16);
      Term _f_26 = _fb_16[0];
      u64 _nd_16 = _sp_16 >= HEAP_OFF ? _sp_16 : heap_alloc(e, 0);
      e.mem[_nd_16 + 0] = rfc_seal(e, _f_26);
      r0 = term_ctr(CID_ir_VE, _nd_16);
      WL_RETN(1);
    } else if (term_aux(_h_31) == CID_ir_VFun) {
      Term _fb_17[2];
      u64 _sp_17 = ctr_take(e, _h_31, 2, _fb_17);
      Term _f_27 = _fb_17[0];
      u32 _f_28 = _fb_17[1];
      _f_27 = term_keep(e, _f_27, 1);
      spare_free(e, cls_fit(2), _sp_17);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_27;
        STK(1) = FID_ir_eval_k47;
        WL_PUSHN(2);
      } else {
        u64 _t_48 = task_node(e, FID_ir_eval_k47, WL_CONT, WL_IDX, 1);
        e.mem[_t_48 + 0] = _f_27;
        WL_CONT = term_tsk(FID_ir_eval_k47, _t_48);
        WL_IDX = 1;
      }
      if (!DEVICE && !seq && fid_nofk(FID_ir_functionKeys)) {
        u64 _t_49 = task_node(e, FID_ir_functionKeys, WL_CONT, WL_IDX, 0);
        e.mem[_t_49 + 0] = _f_27;
        e.mem[_t_49 + 1] = term_pak(CID_Nil, 0);
        return term_tsk(FID_ir_functionKeys, _t_49);
      }
      r0 = _f_27;
      r1 = term_pak(CID_Nil, 0);
      WL_JMP(FID_ir_functionKeys);
    } else {
      term_sink(e, _h_31);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 2258);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_eval_k47)
    WL_POPN(1);
    Term _f_29 = STK(0);
    Term _h_32 = r0;
    WL_OPEN
    Term _o_29[1];
    if (spin_ir_functionSize(e, _o_29, _f_29) == 0) {
      return 0;
    }
    term_sink(e, _f_29);
    if (!DEVICE && !seq && fid_nofk(FID_ir_setSorted)) {
      u64 _t_50 = task_node(e, FID_ir_setSorted, WL_CONT, WL_IDX, 0);
      e.mem[_t_50 + 0] = _h_32;
      e.mem[_t_50 + 1] = (u32)_o_29[0];
      return term_tsk(FID_ir_setSorted, _t_50);
    }
    r0 = _h_32;
    r1 = (u32)_o_29[0];
    WL_JMP(FID_ir_setSorted);
  }}

  WL_CASE(FID_ir_eval_k49)
    WL_POPN(1);
    u32 _o_30 = STK(0);
    Term _h_33 = r0;
    WL_OPEN
    _h_33 = term_keep(e, _h_33, 1);
    if (seq) {
      WL_ROOM(3);
      STK(0) = _o_30;
      STK(1) = _h_33;
      STK(2) = FID_ir_eval_k51;
      WL_PUSHN(3);
    } else {
      u64 _t_53 = task_node(e, FID_ir_eval_k51, WL_CONT, WL_IDX, 1);
      e.mem[_t_53 + 0] = _o_30;
      e.mem[_t_53 + 1] = _h_33;
      WL_CONT = term_tsk(FID_ir_eval_k51, _t_53);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_firstErr)) {
      u64 _t_54 = task_node(e, FID_ir_firstErr, WL_CONT, WL_IDX, 0);
      e.mem[_t_54 + 0] = _h_33;
      return term_tsk(FID_ir_firstErr, _t_54);
    }
    r0 = _h_33;
    WL_JMP(FID_ir_firstErr);
  }}

  WL_CASE(FID_ir_eval_k51)
    WL_POPN(2);
    u32 _o_31 = STK(0);
    Term _h_36 = STK(1);
    u32 _h_34 = r0;
    Term _h_35 = r1;
    WL_OPEN
    if (_h_34 == 1) {
      term_sink(e, _h_36);
      u64 _nd_17 = heap_alloc(e, 0);
      e.mem[_nd_17 + 0] = rfc_seal(e, _h_35);
      r0 = term_ctr(CID_ir_VE, _nd_17);
      WL_RETN(1);
    } else {
      if (_o_31 == 1) {
        Term _o_32[1];
        if (spin_ir_sequenceLengthArg(e, _o_32, _h_36) == 0) {
          return 0;
        }
        r0 = _o_32[0];
        WL_RETN(1);
      } else if ((_o_31 & 7) == 1) {
        term_sink(e, _h_36);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 927);
        WL_RETN(1);
      } else if (_o_31 == 5) {
        if (term_aux(_h_36) == CID_Con) {
          Term _fb_18[2];
          u64 _sp_18 = ctr_take(e, _h_36, 2, _fb_18);
          Term _f_30 = _fb_18[0];
          Term _f_31 = _fb_18[1];
          if (term_aux(_f_30) == CID_ir_VFun) {
            Term _fb_19[2];
            u64 _sp_19 = ctr_take(e, _f_30, 2, _fb_19);
            Term _f_32 = _fb_19[0];
            u32 _f_33 = _fb_19[1];
            if (_f_33 == 1) {
              if (term_aux(_f_31) == CID_Con) {
                Term _fb_20[2];
                u64 _sp_20 = ctr_take(e, _f_31, 2, _fb_20);
                Term _f_34 = _fb_20[0];
                Term _f_35 = _fb_20[1];
                if (term_aux(_f_34) == CID_ir_VFun) {
                  Term _fb_21[2];
                  u64 _sp_21 = ctr_take(e, _f_34, 2, _fb_21);
                  Term _f_36 = _fb_21[0];
                  u32 _f_37 = _fb_21[1];
                  if (_f_37 == 1) {
                    if (term_aux(_f_35) == CID_Nil) {
                      Term _o_33[1];
                      if (spin_ir_functionSize(e, _o_33, _f_32) == 0) {
                        return 0;
                      }
                      Term _o_34[1];
                      if (spin_ir_functionSize(e, _o_34, _f_36) == 0) {
                        return 0;
                      }
                      u32 _s_1 = U32_BIN((u32)_o_33[0], ==, 0ull);
                      if (_s_1 == 1) {
                        u32 _s_2 = U32_BIN((u32)_o_34[0], ==, 0ull);
                        if (_s_2 == 1) {
                          term_sink(e, _f_32);
                          u64 _nd_18 = _sp_18 >= HEAP_OFF ? _sp_18 : heap_alloc(e, 1);
                          e.mem[_nd_18 + 0] = rfc_seal(e, _f_36);
                          e.mem[_nd_18 + 1] = rfc_seal(e, 1);
                          spare_free(e, cls_fit(2), _sp_21);
                          spare_free(e, cls_fit(2), _sp_20);
                          spare_free(e, cls_fit(2), _sp_19);
                          r0 = term_ctr(CID_ir_VFun, _nd_18);
                          WL_RETN(1);
                        } else {
                          term_sink(e, _f_32);
                          u64 _nd_19 = _sp_18 >= HEAP_OFF ? _sp_18 : heap_alloc(e, 1);
                          e.mem[_nd_19 + 0] = rfc_seal(e, _f_36);
                          e.mem[_nd_19 + 1] = rfc_seal(e, 1);
                          spare_free(e, cls_fit(2), _sp_21);
                          spare_free(e, cls_fit(2), _sp_20);
                          spare_free(e, cls_fit(2), _sp_19);
                          r0 = term_ctr(CID_ir_VFun, _nd_19);
                          WL_RETN(1);
                        }
                      } else {
                        u32 _s_3 = U32_BIN((u32)_o_34[0], ==, 0ull);
                        if (_s_3 == 1) {
                          term_sink(e, _f_36);
                          u64 _nd_20 = _sp_18 >= HEAP_OFF ? _sp_18 : heap_alloc(e, 1);
                          e.mem[_nd_20 + 0] = rfc_seal(e, _f_32);
                          e.mem[_nd_20 + 1] = rfc_seal(e, 1);
                          spare_free(e, cls_fit(2), _sp_21);
                          spare_free(e, cls_fit(2), _sp_20);
                          spare_free(e, cls_fit(2), _sp_19);
                          r0 = term_ctr(CID_ir_VFun, _nd_20);
                          WL_RETN(1);
                        } else {
                          Term _a_0 = U32_BIN(1048576ull, -, (u32)_o_34[0]);
                          u32 _s_4 = U32_BIN((u32)_o_33[0], <=, _a_0);
                          if (_s_4 == 0) {
                            term_sink(e, _f_32);
                            term_sink(e, _f_36);
                            spare_free(e, cls_fit(2), _sp_21);
                            spare_free(e, cls_fit(2), _sp_20);
                            spare_free(e, cls_fit(2), _sp_19);
                            spare_free(e, cls_fit(2), _sp_18);
                            r0 = term_ctr(CID_ir_VE, STAT_OFF + 92);
                            WL_RETN(1);
                          } else {
                            spare_free(e, cls_fit(2), _sp_21);
                            spare_free(e, cls_fit(2), _sp_20);
                            spare_free(e, cls_fit(2), _sp_19);
                            spare_free(e, cls_fit(2), _sp_18);
                            if (seq) {
                              WL_ROOM(2);
                              STK(0) = _f_32;
                              STK(1) = FID_ir_eval_k53;
                              WL_PUSHN(2);
                            } else {
                              u64 _t_55 = task_node(e, FID_ir_eval_k53, WL_CONT, WL_IDX, 1);
                              e.mem[_t_55 + 0] = _f_32;
                              WL_CONT = term_tsk(FID_ir_eval_k53, _t_55);
                              WL_IDX = 1;
                            }
                            if (!DEVICE && !seq && fid_nofk(FID_ir_functionValues)) {
                              u64 _t_56 = task_node(e, FID_ir_functionValues, WL_CONT, WL_IDX, 0);
                              e.mem[_t_56 + 0] = _f_36;
                              e.mem[_t_56 + 1] = term_pak(CID_Nil, 0);
                              return term_tsk(FID_ir_functionValues, _t_56);
                            }
                            r0 = _f_36;
                            r1 = term_pak(CID_Nil, 0);
                            WL_JMP(FID_ir_functionValues);
                          }
                        }
                      }
                    } else {
                      term_sink(e, _f_32);
                      term_sink(e, _f_36);
                      term_sink(e, _f_35);
                      spare_free(e, cls_fit(2), _sp_21);
                      spare_free(e, cls_fit(2), _sp_20);
                      spare_free(e, cls_fit(2), _sp_19);
                      spare_free(e, cls_fit(2), _sp_18);
                      r0 = term_ctr(CID_ir_VE, STAT_OFF + 784);
                      WL_RETN(1);
                    }
                  } else {
                    term_sink(e, _f_32);
                    term_sink(e, _f_36);
                    term_sink(e, _f_35);
                    spare_free(e, cls_fit(2), _sp_21);
                    spare_free(e, cls_fit(2), _sp_20);
                    spare_free(e, cls_fit(2), _sp_19);
                    spare_free(e, cls_fit(2), _sp_18);
                    r0 = term_ctr(CID_ir_VE, STAT_OFF + 784);
                    WL_RETN(1);
                  }
                } else {
                  term_sink(e, _f_32);
                  term_sink(e, _f_34);
                  term_sink(e, _f_35);
                  spare_free(e, cls_fit(2), _sp_20);
                  spare_free(e, cls_fit(2), _sp_19);
                  spare_free(e, cls_fit(2), _sp_18);
                  r0 = term_ctr(CID_ir_VE, STAT_OFF + 784);
                  WL_RETN(1);
                }
              } else {
                term_sink(e, _f_32);
                term_sink(e, _f_31);
                spare_free(e, cls_fit(2), _sp_19);
                spare_free(e, cls_fit(2), _sp_18);
                r0 = term_ctr(CID_ir_VE, STAT_OFF + 784);
                WL_RETN(1);
              }
            } else {
              term_sink(e, _f_32);
              term_sink(e, _f_31);
              spare_free(e, cls_fit(2), _sp_19);
              spare_free(e, cls_fit(2), _sp_18);
              r0 = term_ctr(CID_ir_VE, STAT_OFF + 784);
              WL_RETN(1);
            }
          } else {
            term_sink(e, _f_30);
            term_sink(e, _f_31);
            spare_free(e, cls_fit(2), _sp_18);
            r0 = term_ctr(CID_ir_VE, STAT_OFF + 784);
            WL_RETN(1);
          }
        } else {
          term_sink(e, _h_36);
          r0 = term_ctr(CID_ir_VE, STAT_OFF + 784);
          WL_RETN(1);
        }
      } else if ((_o_31 & 7) == 5) {
        term_sink(e, _h_36);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 927);
        WL_RETN(1);
      } else if (_o_31 == 3) {
        if (term_aux(_h_36) == CID_Con) {
          Term _fb_22[2];
          u64 _sp_22 = ctr_take(e, _h_36, 2, _fb_22);
          Term _f_39 = _fb_22[0];
          Term _f_40 = _fb_22[1];
          if (term_aux(_f_39) == CID_ir_VFun) {
            Term _fb_23[2];
            u64 _sp_23 = ctr_take(e, _f_39, 2, _fb_23);
            Term _f_41 = _fb_23[0];
            u32 _f_42 = _fb_23[1];
            if (_f_42 == 1) {
              if (term_aux(_f_40) == CID_Nil) {
                Term _o_35[1];
                if (spin_ir_functionSize(e, _o_35, _f_41) == 0) {
                  return 0;
                }
                if ((u32)_o_35[0] == 0) {
                  term_sink(e, _f_41);
                  spare_free(e, cls_fit(2), _sp_23);
                  spare_free(e, cls_fit(2), _sp_22);
                  r0 = term_ctr(CID_ir_VE, STAT_OFF + 362);
                  WL_RETN(1);
                } else {
                  spare_free(e, cls_fit(2), _sp_23);
                  spare_free(e, cls_fit(2), _sp_22);
                  if (seq) {
                    WL_ROOM(1);
                    STK(0) = FID_ir_eval_k57;
                    WL_PUSHN(1);
                  } else {
                    u64 _t_60 = task_node(e, FID_ir_eval_k57, WL_CONT, WL_IDX, 1);
                    WL_CONT = term_tsk(FID_ir_eval_k57, _t_60);
                    WL_IDX = 0;
                  }
                  if (!DEVICE && !seq && fid_nofk(FID_ir_functionValues)) {
                    u64 _t_61 = task_node(e, FID_ir_functionValues, WL_CONT, WL_IDX, 0);
                    e.mem[_t_61 + 0] = _f_41;
                    e.mem[_t_61 + 1] = term_pak(CID_Nil, 0);
                    return term_tsk(FID_ir_functionValues, _t_61);
                  }
                  r0 = _f_41;
                  r1 = term_pak(CID_Nil, 0);
                  WL_JMP(FID_ir_functionValues);
                }
              } else {
                term_sink(e, _f_41);
                term_sink(e, _f_40);
                spare_free(e, cls_fit(2), _sp_23);
                spare_free(e, cls_fit(2), _sp_22);
                r0 = term_ctr(CID_ir_VE, STAT_OFF + 719);
                WL_RETN(1);
              }
            } else {
              term_sink(e, _f_41);
              term_sink(e, _f_40);
              spare_free(e, cls_fit(2), _sp_23);
              spare_free(e, cls_fit(2), _sp_22);
              r0 = term_ctr(CID_ir_VE, STAT_OFF + 719);
              WL_RETN(1);
            }
          } else {
            term_sink(e, _f_39);
            term_sink(e, _f_40);
            spare_free(e, cls_fit(2), _sp_22);
            r0 = term_ctr(CID_ir_VE, STAT_OFF + 719);
            WL_RETN(1);
          }
        } else {
          term_sink(e, _h_36);
          r0 = term_ctr(CID_ir_VE, STAT_OFF + 719);
          WL_RETN(1);
        }
      } else if ((_o_31 & 3) == 3) {
        term_sink(e, _h_36);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 927);
        WL_RETN(1);
      } else if (_o_31 == 2) {
        if (term_aux(_h_36) == CID_Con) {
          Term _fb_24[2];
          u64 _sp_24 = ctr_take(e, _h_36, 2, _fb_24);
          Term _f_43 = _fb_24[0];
          Term _f_44 = _fb_24[1];
          if (term_aux(_f_43) == CID_ir_VFun) {
            Term _fb_25[2];
            u64 _sp_25 = ctr_take(e, _f_43, 2, _fb_25);
            Term _f_45 = _fb_25[0];
            u32 _f_46 = _fb_25[1];
            if (_f_46 == 1) {
              if (term_aux(_f_44) == CID_Nil) {
                spare_free(e, cls_fit(2), _sp_25);
                spare_free(e, cls_fit(2), _sp_24);
                if (seq) {
                  WL_ROOM(1);
                  STK(0) = FID_ir_eval_k59;
                  WL_PUSHN(1);
                } else {
                  u64 _t_63 = task_node(e, FID_ir_eval_k59, WL_CONT, WL_IDX, 1);
                  WL_CONT = term_tsk(FID_ir_eval_k59, _t_63);
                  WL_IDX = 0;
                }
                if (!DEVICE && !seq && fid_nofk(FID_ir_functionFind)) {
                  u64 _t_64 = task_node(e, FID_ir_functionFind, WL_CONT, WL_IDX, 0);
                  e.mem[_t_64 + 0] = _f_45;
                  e.mem[_t_64 + 1] = term_pak(CID_ir_VI, 1ull);
                  return term_tsk(FID_ir_functionFind, _t_64);
                }
                r0 = _f_45;
                r1 = term_pak(CID_ir_VI, 1ull);
                WL_JMP(FID_ir_functionFind);
              } else {
                term_sink(e, _f_45);
                term_sink(e, _f_44);
                spare_free(e, cls_fit(2), _sp_25);
                spare_free(e, cls_fit(2), _sp_24);
                r0 = term_ctr(CID_ir_VE, STAT_OFF + 710);
                WL_RETN(1);
              }
            } else {
              term_sink(e, _f_45);
              term_sink(e, _f_44);
              spare_free(e, cls_fit(2), _sp_25);
              spare_free(e, cls_fit(2), _sp_24);
              r0 = term_ctr(CID_ir_VE, STAT_OFF + 710);
              WL_RETN(1);
            }
          } else {
            term_sink(e, _f_43);
            term_sink(e, _f_44);
            spare_free(e, cls_fit(2), _sp_24);
            r0 = term_ctr(CID_ir_VE, STAT_OFF + 710);
            WL_RETN(1);
          }
        } else {
          term_sink(e, _h_36);
          r0 = term_ctr(CID_ir_VE, STAT_OFF + 710);
          WL_RETN(1);
        }
      } else if ((_o_31 & 7) == 2) {
        term_sink(e, _h_36);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 927);
        WL_RETN(1);
      } else if (_o_31 == 6) {
        if (term_aux(_h_36) == CID_Con) {
          Term _fb_26[2];
          u64 _sp_26 = ctr_take(e, _h_36, 2, _fb_26);
          Term _f_47 = _fb_26[0];
          Term _f_48 = _fb_26[1];
          if (term_aux(_f_47) == CID_ir_VFun) {
            Term _fb_27[2];
            u64 _sp_27 = ctr_take(e, _f_47, 2, _fb_27);
            Term _f_49 = _fb_27[0];
            u32 _f_50 = _fb_27[1];
            if (_f_50 == 1) {
              if (term_aux(_f_48) == CID_Con) {
                Term _fb_28[2];
                u64 _sp_28 = ctr_take(e, _f_48, 2, _fb_28);
                Term _f_51 = _fb_28[0];
                Term _f_52 = _fb_28[1];
                if (term_aux(_f_51) == CID_ir_VI) {
                  u32 _first_0 = term_loc(_f_51);
                  if (term_aux(_f_52) == CID_Con) {
                    Term _fb_29[2];
                    u64 _sp_29 = ctr_take(e, _f_52, 2, _fb_29);
                    Term _f_53 = _fb_29[0];
                    Term _f_54 = _fb_29[1];
                    if (term_aux(_f_53) == CID_ir_VI) {
                      u32 _last_0 = term_loc(_f_53);
                      if (term_aux(_f_54) == CID_Nil) {
                        Term _o_38[1];
                        if (spin_ir_sgt(e, _o_38, _first_0, _last_0) == 0) {
                          return 0;
                        }
                        if ((u32)_o_38[0] == 1) {
                          term_sink(e, _f_49);
                          Term _o_39[1];
                          if (spin_ir_emptyFunction(e, _o_39) == 0) {
                            return 0;
                          }
                          spare_free(e, cls_fit(2), _sp_29);
                          spare_free(e, cls_fit(2), _sp_28);
                          spare_free(e, cls_fit(2), _sp_27);
                          spare_free(e, cls_fit(2), _sp_26);
                          r0 = _o_39[0];
                          WL_RETN(1);
                        } else {
                          Term _o_40[1];
                          if (spin_ir_sge(e, _o_40, _first_0, 1ull) == 0) {
                            return 0;
                          }
                          Term _o_41[1];
                          if (spin_ir_functionSize(e, _o_41, _f_49) == 0) {
                            return 0;
                          }
                          Term _o_42[1];
                          if (spin_ir_sle(e, _o_42, _last_0, (u32)_o_41[0]) == 0) {
                            return 0;
                          }
                          Term _o_43[1];
                          if (spin_Bool_and(e, _o_43, (u32)_o_40[0], (u32)_o_42[0]) == 0) {
                            return 0;
                          }
                          if ((u32)_o_43[0] == 0) {
                            term_sink(e, _f_49);
                            spare_free(e, cls_fit(2), _sp_29);
                            spare_free(e, cls_fit(2), _sp_28);
                            spare_free(e, cls_fit(2), _sp_27);
                            spare_free(e, cls_fit(2), _sp_26);
                            r0 = term_ctr(CID_ir_VE, STAT_OFF + 159);
                            WL_RETN(1);
                          } else {
                            Term _o_44[1];
                            if (spin_ir_functionSize(e, _o_44, _f_49) == 0) {
                              return 0;
                            }
                            Term _o_45[1];
                            if (spin_Bool_and(e, _o_45, U32_BIN(_first_0, ==, 1ull), U32_BIN(_last_0, ==, (u32)_o_44[0])) == 0) {
                              return 0;
                            }
                            if ((u32)_o_45[0] == 1) {
                              u64 _nd_21 = _sp_26 >= HEAP_OFF ? _sp_26 : heap_alloc(e, 1);
                              e.mem[_nd_21 + 0] = rfc_seal(e, _f_49);
                              e.mem[_nd_21 + 1] = rfc_seal(e, 1);
                              spare_free(e, cls_fit(2), _sp_29);
                              spare_free(e, cls_fit(2), _sp_28);
                              spare_free(e, cls_fit(2), _sp_27);
                              r0 = term_ctr(CID_ir_VFun, _nd_21);
                              WL_RETN(1);
                            } else {
                              spare_free(e, cls_fit(2), _sp_29);
                              spare_free(e, cls_fit(2), _sp_28);
                              spare_free(e, cls_fit(2), _sp_27);
                              spare_free(e, cls_fit(2), _sp_26);
                              if (seq) {
                                WL_ROOM(3);
                                STK(0) = _first_0;
                                STK(1) = _last_0;
                                STK(2) = FID_ir_eval_k61;
                                WL_PUSHN(3);
                              } else {
                                u64 _t_65 = task_node(e, FID_ir_eval_k61, WL_CONT, WL_IDX, 1);
                                e.mem[_t_65 + 0] = _first_0;
                                e.mem[_t_65 + 1] = _last_0;
                                WL_CONT = term_tsk(FID_ir_eval_k61, _t_65);
                                WL_IDX = 2;
                              }
                              if (!DEVICE && !seq && fid_nofk(FID_ir_functionValues)) {
                                u64 _t_66 = task_node(e, FID_ir_functionValues, WL_CONT, WL_IDX, 0);
                                e.mem[_t_66 + 0] = _f_49;
                                e.mem[_t_66 + 1] = term_pak(CID_Nil, 0);
                                return term_tsk(FID_ir_functionValues, _t_66);
                              }
                              r0 = _f_49;
                              r1 = term_pak(CID_Nil, 0);
                              WL_JMP(FID_ir_functionValues);
                            }
                          }
                        }
                      } else {
                        term_sink(e, _f_49);
                        term_sink(e, _f_54);
                        spare_free(e, cls_fit(2), _sp_29);
                        spare_free(e, cls_fit(2), _sp_28);
                        spare_free(e, cls_fit(2), _sp_27);
                        spare_free(e, cls_fit(2), _sp_26);
                        r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
                        WL_RETN(1);
                      }
                    } else {
                      term_sink(e, _f_49);
                      term_sink(e, _f_53);
                      term_sink(e, _f_54);
                      spare_free(e, cls_fit(2), _sp_29);
                      spare_free(e, cls_fit(2), _sp_28);
                      spare_free(e, cls_fit(2), _sp_27);
                      spare_free(e, cls_fit(2), _sp_26);
                      r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
                      WL_RETN(1);
                    }
                  } else {
                    term_sink(e, _f_49);
                    term_sink(e, _f_52);
                    spare_free(e, cls_fit(2), _sp_28);
                    spare_free(e, cls_fit(2), _sp_27);
                    spare_free(e, cls_fit(2), _sp_26);
                    r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
                    WL_RETN(1);
                  }
                } else {
                  term_sink(e, _f_49);
                  term_sink(e, _f_51);
                  term_sink(e, _f_52);
                  spare_free(e, cls_fit(2), _sp_28);
                  spare_free(e, cls_fit(2), _sp_27);
                  spare_free(e, cls_fit(2), _sp_26);
                  r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
                  WL_RETN(1);
                }
              } else {
                term_sink(e, _f_49);
                term_sink(e, _f_48);
                spare_free(e, cls_fit(2), _sp_27);
                spare_free(e, cls_fit(2), _sp_26);
                r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
                WL_RETN(1);
              }
            } else {
              term_sink(e, _f_49);
              term_sink(e, _f_48);
              spare_free(e, cls_fit(2), _sp_27);
              spare_free(e, cls_fit(2), _sp_26);
              r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
              WL_RETN(1);
            }
          } else {
            term_sink(e, _f_47);
            term_sink(e, _f_48);
            spare_free(e, cls_fit(2), _sp_26);
            r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
            WL_RETN(1);
          }
        } else {
          term_sink(e, _h_36);
          r0 = term_ctr(CID_ir_VE, STAT_OFF + 669);
          WL_RETN(1);
        }
      } else if ((_o_31 & 7) == 6) {
        term_sink(e, _h_36);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 927);
        WL_RETN(1);
      } else if (_o_31 == 4) {
        if (term_aux(_h_36) == CID_Con) {
          Term _fb_30[2];
          u64 _sp_30 = ctr_take(e, _h_36, 2, _fb_30);
          Term _f_55 = _fb_30[0];
          Term _f_56 = _fb_30[1];
          if (term_aux(_f_55) == CID_ir_VFun) {
            Term _fb_31[2];
            u64 _sp_31 = ctr_take(e, _f_55, 2, _fb_31);
            Term _f_57 = _fb_31[0];
            u32 _f_58 = _fb_31[1];
            if (_f_58 == 1) {
              if (term_aux(_f_56) == CID_Con) {
                Term _fb_32[2];
                u64 _sp_32 = ctr_take(e, _f_56, 2, _fb_32);
                Term _f_59 = _fb_32[0];
                Term _f_60 = _fb_32[1];
                if (term_aux(_f_60) == CID_Nil) {
                  Term _o_47[1];
                  if (spin_ir_functionSize(e, _o_47, _f_57) == 0) {
                    return 0;
                  }
                  u32 _s_5 = U32_BIN((u32)_o_47[0], <, 1048576ull);
                  if (_s_5 == 0) {
                    term_sink(e, _f_57);
                    term_sink(e, _f_59);
                    spare_free(e, cls_fit(2), _sp_32);
                    spare_free(e, cls_fit(2), _sp_31);
                    spare_free(e, cls_fit(2), _sp_30);
                    r0 = term_ctr(CID_ir_VE, STAT_OFF + 92);
                    WL_RETN(1);
                  } else {
                    Term _o_48[1];
                    if (spin_ir_functionSize(e, _o_48, _f_57) == 0) {
                      return 0;
                    }
                    spare_free(e, cls_fit(2), _sp_32);
                    spare_free(e, cls_fit(2), _sp_31);
                    spare_free(e, cls_fit(2), _sp_30);
                    if (seq) {
                      WL_ROOM(1);
                      STK(0) = FID_ir_eval_k65;
                      WL_PUSHN(1);
                    } else {
                      u64 _t_70 = task_node(e, FID_ir_eval_k65, WL_CONT, WL_IDX, 1);
                      WL_CONT = term_tsk(FID_ir_eval_k65, _t_70);
                      WL_IDX = 0;
                    }
                    if (!DEVICE && !seq && fid_nofk(FID_ir_functionAppendTree)) {
                      u64 _t_71 = task_node(e, FID_ir_functionAppendTree, WL_CONT, WL_IDX, 0);
                      e.mem[_t_71 + 0] = _f_57;
                      e.mem[_t_71 + 1] = term_pak(CID_ir_VI, U32_BIN((u32)_o_48[0], +, 1));
                      e.mem[_t_71 + 2] = _f_59;
                      return term_tsk(FID_ir_functionAppendTree, _t_71);
                    }
                    r0 = _f_57;
                    r1 = term_pak(CID_ir_VI, U32_BIN((u32)_o_48[0], +, 1));
                    r2 = _f_59;
                    WL_JMP(FID_ir_functionAppendTree);
                  }
                } else {
                  term_sink(e, _f_57);
                  term_sink(e, _f_59);
                  term_sink(e, _f_60);
                  spare_free(e, cls_fit(2), _sp_32);
                  spare_free(e, cls_fit(2), _sp_31);
                  spare_free(e, cls_fit(2), _sp_30);
                  r0 = term_ctr(CID_ir_VE, STAT_OFF + 578);
                  WL_RETN(1);
                }
              } else {
                term_sink(e, _f_57);
                term_sink(e, _f_56);
                spare_free(e, cls_fit(2), _sp_31);
                spare_free(e, cls_fit(2), _sp_30);
                r0 = term_ctr(CID_ir_VE, STAT_OFF + 578);
                WL_RETN(1);
              }
            } else {
              term_sink(e, _f_57);
              term_sink(e, _f_56);
              spare_free(e, cls_fit(2), _sp_31);
              spare_free(e, cls_fit(2), _sp_30);
              r0 = term_ctr(CID_ir_VE, STAT_OFF + 578);
              WL_RETN(1);
            }
          } else {
            term_sink(e, _f_55);
            term_sink(e, _f_56);
            spare_free(e, cls_fit(2), _sp_30);
            r0 = term_ctr(CID_ir_VE, STAT_OFF + 578);
            WL_RETN(1);
          }
        } else {
          term_sink(e, _h_36);
          r0 = term_ctr(CID_ir_VE, STAT_OFF + 578);
          WL_RETN(1);
        }
      } else {
        term_sink(e, _h_36);
        r0 = term_ctr(CID_ir_VE, STAT_OFF + 927);
        WL_RETN(1);
      }
    }
  }}

  WL_CASE(FID_ir_eval_k53)
    WL_POPN(1);
    Term _f_38 = STK(0);
    Term _h_37 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(1);
      STK(0) = FID_ir_eval_k55;
      WL_PUSHN(1);
    } else {
      u64 _t_57 = task_node(e, FID_ir_eval_k55, WL_CONT, WL_IDX, 1);
      WL_CONT = term_tsk(FID_ir_eval_k55, _t_57);
      WL_IDX = 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_functionValues)) {
      u64 _t_58 = task_node(e, FID_ir_functionValues, WL_CONT, WL_IDX, 0);
      e.mem[_t_58 + 0] = _f_38;
      e.mem[_t_58 + 1] = _h_37;
      return term_tsk(FID_ir_functionValues, _t_58);
    }
    r0 = _f_38;
    r1 = _h_37;
    WL_JMP(FID_ir_functionValues);
  }}

  WL_CASE(FID_ir_eval_k55)
    Term _h_38 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_sequenceValue)) {
      u64 _t_59 = task_node(e, FID_ir_sequenceValue, WL_CONT, WL_IDX, 0);
      e.mem[_t_59 + 0] = _h_38;
      return term_tsk(FID_ir_sequenceValue, _t_59);
    }
    r0 = _h_38;
    WL_JMP(FID_ir_sequenceValue);
  }}

  WL_CASE(FID_ir_eval_k57)
    Term _h_39 = r0;
    WL_OPEN
    Term _o_36[1];
    if (spin_ir_sequenceDrop(e, _o_36, _h_39, 1ull) == 0) {
      return 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_sequenceValue)) {
      u64 _t_62 = task_node(e, FID_ir_sequenceValue, WL_CONT, WL_IDX, 0);
      e.mem[_t_62 + 0] = _o_36[0];
      return term_tsk(FID_ir_sequenceValue, _t_62);
    }
    r0 = _o_36[0];
    WL_JMP(FID_ir_sequenceValue);
  }}

  WL_CASE(FID_ir_eval_k59)
    u32 _h_40 = r0;
    Term _h_41 = r1;
    WL_OPEN
    Term _o_37[1];
    if (spin_ir_functionFound(e, _o_37, _h_40, _h_41) == 0) {
      return 0;
    }
    r0 = _o_37[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k61)
    WL_POPN(2);
    u32 _first_1 = STK(0);
    u32 _last_1 = STK(1);
    Term _h_42 = r0;
    WL_OPEN
    Term _a_1 = U32_BIN(_first_1, -, 1ull);
    Term _o_46[1];
    if (spin_ir_sequenceDrop(e, _o_46, _h_42, ((u64)(_a_1))) == 0) {
      return 0;
    }
    Term _a_2 = U32_BIN(_last_1, -, _first_1);
    Term _a_3 = U32_BIN(_a_2, +, 1);
    if (seq) {
      WL_ROOM(1);
      STK(0) = FID_ir_eval_k63;
      WL_PUSHN(1);
    } else {
      u64 _t_67 = task_node(e, FID_ir_eval_k63, WL_CONT, WL_IDX, 1);
      WL_CONT = term_tsk(FID_ir_eval_k63, _t_67);
      WL_IDX = 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_sequenceTake)) {
      u64 _t_68 = task_node(e, FID_ir_sequenceTake, WL_CONT, WL_IDX, 0);
      e.mem[_t_68 + 0] = _o_46[0];
      e.mem[_t_68 + 1] = ((u64)(_a_3));
      return term_tsk(FID_ir_sequenceTake, _t_68);
    }
    r0 = _o_46[0];
    r1 = ((u64)(_a_3));
    WL_JMP(FID_ir_sequenceTake);
  }}

  WL_CASE(FID_ir_eval_k63)
    Term _h_43 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_sequenceValue)) {
      u64 _t_69 = task_node(e, FID_ir_sequenceValue, WL_CONT, WL_IDX, 0);
      e.mem[_t_69 + 0] = _h_43;
      return term_tsk(FID_ir_sequenceValue, _t_69);
    }
    r0 = _h_43;
    WL_JMP(FID_ir_sequenceValue);
  }}

  WL_CASE(FID_ir_eval_k65)
    Term _h_44 = r0;
    WL_OPEN
    u64 _nd_22 = heap_alloc(e, 1);
    e.mem[_nd_22 + 0] = rfc_seal(e, _h_44);
    e.mem[_nd_22 + 1] = rfc_seal(e, 1);
    r0 = term_ctr(CID_ir_VFun, _nd_22);
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k67)
    WL_POPN(4);
    Term _nodes_7 = STK(0);
    Term _env_17 = STK(1);
    Term _env_18 = STK(2);
    u32 _o_49 = STK(3);
    Term _h_45 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_45;
      STK(1) = FID_ir_eval_k69;
      WL_PUSHN(2);
    } else {
      u64 _t_73 = task_node(e, FID_ir_eval_k69, WL_CONT, WL_IDX, 1);
      e.mem[_t_73 + 0] = _h_45;
      WL_CONT = term_tsk(FID_ir_eval_k69, _t_73);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_74 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_74 + 0] = _nodes_7;
      e.mem[_t_74 + 1] = _env_17;
      e.mem[_t_74 + 2] = _env_18;
      e.mem[_t_74 + 3] = _o_49;
      return term_tsk(FID_ir_eval, _t_74);
    }
    r0 = _nodes_7;
    r1 = _env_17;
    r2 = _env_18;
    r3 = _o_49;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k69)
    WL_POPN(1);
    Term _h_47 = STK(0);
    Term _h_46 = r0;
    WL_OPEN
    Term _o_50[1];
    if (spin_ir_vadd(e, _o_50, _h_47, _h_46) == 0) {
      return 0;
    }
    r0 = _o_50[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k71)
    WL_POPN(4);
    Term _nodes_8 = STK(0);
    Term _env_19 = STK(1);
    Term _env_20 = STK(2);
    u32 _o_51 = STK(3);
    Term _h_48 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_48;
      STK(1) = FID_ir_eval_k73;
      WL_PUSHN(2);
    } else {
      u64 _t_76 = task_node(e, FID_ir_eval_k73, WL_CONT, WL_IDX, 1);
      e.mem[_t_76 + 0] = _h_48;
      WL_CONT = term_tsk(FID_ir_eval_k73, _t_76);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_77 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_77 + 0] = _nodes_8;
      e.mem[_t_77 + 1] = _env_19;
      e.mem[_t_77 + 2] = _env_20;
      e.mem[_t_77 + 3] = _o_51;
      return term_tsk(FID_ir_eval, _t_77);
    }
    r0 = _nodes_8;
    r1 = _env_19;
    r2 = _env_20;
    r3 = _o_51;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k73)
    WL_POPN(1);
    Term _h_50 = STK(0);
    Term _h_49 = r0;
    WL_OPEN
    Term _o_52[1];
    if (spin_ir_vsub(e, _o_52, _h_50, _h_49) == 0) {
      return 0;
    }
    r0 = _o_52[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k75)
    Term _h_51 = r0;
    WL_OPEN
    Term _o_53[1];
    if (spin_ir_vneg(e, _o_53, _h_51) == 0) {
      return 0;
    }
    r0 = _o_53[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k77)
    WL_POPN(4);
    Term _nodes_9 = STK(0);
    Term _env_21 = STK(1);
    Term _env_22 = STK(2);
    u32 _o_54 = STK(3);
    Term _h_52 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_52;
      STK(1) = FID_ir_eval_k79;
      WL_PUSHN(2);
    } else {
      u64 _t_80 = task_node(e, FID_ir_eval_k79, WL_CONT, WL_IDX, 1);
      e.mem[_t_80 + 0] = _h_52;
      WL_CONT = term_tsk(FID_ir_eval_k79, _t_80);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_81 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_81 + 0] = _nodes_9;
      e.mem[_t_81 + 1] = _env_21;
      e.mem[_t_81 + 2] = _env_22;
      e.mem[_t_81 + 3] = _o_54;
      return term_tsk(FID_ir_eval, _t_81);
    }
    r0 = _nodes_9;
    r1 = _env_21;
    r2 = _env_22;
    r3 = _o_54;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k79)
    WL_POPN(1);
    Term _h_54 = STK(0);
    Term _h_53 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_veqb)) {
      u64 _t_82 = task_node(e, FID_ir_veqb, WL_CONT, WL_IDX, 0);
      e.mem[_t_82 + 0] = _h_54;
      e.mem[_t_82 + 1] = _h_53;
      return term_tsk(FID_ir_veqb, _t_82);
    }
    r0 = _h_54;
    r1 = _h_53;
    WL_JMP(FID_ir_veqb);
  }}

  WL_CASE(FID_ir_eval_k81)
    WL_POPN(4);
    Term _nodes_10 = STK(0);
    Term _env_23 = STK(1);
    Term _env_24 = STK(2);
    u32 _o_55 = STK(3);
    Term _h_55 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_55;
      STK(1) = FID_ir_eval_k83;
      WL_PUSHN(2);
    } else {
      u64 _t_84 = task_node(e, FID_ir_eval_k83, WL_CONT, WL_IDX, 1);
      e.mem[_t_84 + 0] = _h_55;
      WL_CONT = term_tsk(FID_ir_eval_k83, _t_84);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_85 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_85 + 0] = _nodes_10;
      e.mem[_t_85 + 1] = _env_23;
      e.mem[_t_85 + 2] = _env_24;
      e.mem[_t_85 + 3] = _o_55;
      return term_tsk(FID_ir_eval, _t_85);
    }
    r0 = _nodes_10;
    r1 = _env_23;
    r2 = _env_24;
    r3 = _o_55;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k83)
    WL_POPN(1);
    Term _h_57 = STK(0);
    Term _h_56 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(1);
      STK(0) = FID_ir_eval_k85;
      WL_PUSHN(1);
    } else {
      u64 _t_86 = task_node(e, FID_ir_eval_k85, WL_CONT, WL_IDX, 1);
      WL_CONT = term_tsk(FID_ir_eval_k85, _t_86);
      WL_IDX = 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_veqb)) {
      u64 _t_87 = task_node(e, FID_ir_veqb, WL_CONT, WL_IDX, 0);
      e.mem[_t_87 + 0] = _h_57;
      e.mem[_t_87 + 1] = _h_56;
      return term_tsk(FID_ir_veqb, _t_87);
    }
    r0 = _h_57;
    r1 = _h_56;
    WL_JMP(FID_ir_veqb);
  }}

  WL_CASE(FID_ir_eval_k85)
    Term _h_58 = r0;
    WL_OPEN
    Term _o_56[1];
    if (spin_ir_vnot(e, _o_56, _h_58) == 0) {
      return 0;
    }
    r0 = _o_56[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k87)
    WL_POPN(4);
    Term _nodes_11 = STK(0);
    Term _env_25 = STK(1);
    Term _env_26 = STK(2);
    u32 _o_57 = STK(3);
    Term _h_59 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_59;
      STK(1) = FID_ir_eval_k89;
      WL_PUSHN(2);
    } else {
      u64 _t_89 = task_node(e, FID_ir_eval_k89, WL_CONT, WL_IDX, 1);
      e.mem[_t_89 + 0] = _h_59;
      WL_CONT = term_tsk(FID_ir_eval_k89, _t_89);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_90 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_90 + 0] = _nodes_11;
      e.mem[_t_90 + 1] = _env_25;
      e.mem[_t_90 + 2] = _env_26;
      e.mem[_t_90 + 3] = _o_57;
      return term_tsk(FID_ir_eval, _t_90);
    }
    r0 = _nodes_11;
    r1 = _env_25;
    r2 = _env_26;
    r3 = _o_57;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k89)
    WL_POPN(1);
    Term _h_61 = STK(0);
    Term _h_60 = r0;
    WL_OPEN
    Term _o_58[1];
    if (spin_ir_vlt(e, _o_58, _h_61, _h_60) == 0) {
      return 0;
    }
    r0 = _o_58[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k91)
    WL_POPN(4);
    Term _nodes_12 = STK(0);
    Term _env_27 = STK(1);
    Term _env_28 = STK(2);
    u32 _o_59 = STK(3);
    Term _h_62 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_62;
      STK(1) = FID_ir_eval_k93;
      WL_PUSHN(2);
    } else {
      u64 _t_92 = task_node(e, FID_ir_eval_k93, WL_CONT, WL_IDX, 1);
      e.mem[_t_92 + 0] = _h_62;
      WL_CONT = term_tsk(FID_ir_eval_k93, _t_92);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_93 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_93 + 0] = _nodes_12;
      e.mem[_t_93 + 1] = _env_27;
      e.mem[_t_93 + 2] = _env_28;
      e.mem[_t_93 + 3] = _o_59;
      return term_tsk(FID_ir_eval, _t_93);
    }
    r0 = _nodes_12;
    r1 = _env_27;
    r2 = _env_28;
    r3 = _o_59;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k93)
    WL_POPN(1);
    Term _h_64 = STK(0);
    Term _h_63 = r0;
    WL_OPEN
    Term _o_60[1];
    if (spin_ir_vle(e, _o_60, _h_64, _h_63) == 0) {
      return 0;
    }
    r0 = _o_60[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k95)
    WL_POPN(4);
    Term _nodes_13 = STK(0);
    Term _env_29 = STK(1);
    Term _env_30 = STK(2);
    u32 _o_61 = STK(3);
    Term _h_65 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_65;
      STK(1) = FID_ir_eval_k97;
      WL_PUSHN(2);
    } else {
      u64 _t_95 = task_node(e, FID_ir_eval_k97, WL_CONT, WL_IDX, 1);
      e.mem[_t_95 + 0] = _h_65;
      WL_CONT = term_tsk(FID_ir_eval_k97, _t_95);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_96 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_96 + 0] = _nodes_13;
      e.mem[_t_96 + 1] = _env_29;
      e.mem[_t_96 + 2] = _env_30;
      e.mem[_t_96 + 3] = _o_61;
      return term_tsk(FID_ir_eval, _t_96);
    }
    r0 = _nodes_13;
    r1 = _env_29;
    r2 = _env_30;
    r3 = _o_61;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k97)
    WL_POPN(1);
    Term _h_67 = STK(0);
    Term _h_66 = r0;
    WL_OPEN
    Term _o_62[1];
    if (spin_ir_vgt(e, _o_62, _h_67, _h_66) == 0) {
      return 0;
    }
    r0 = _o_62[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k99)
    WL_POPN(4);
    Term _nodes_14 = STK(0);
    Term _env_31 = STK(1);
    Term _env_32 = STK(2);
    u32 _o_63 = STK(3);
    Term _h_68 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_68;
      STK(1) = FID_ir_eval_k101;
      WL_PUSHN(2);
    } else {
      u64 _t_98 = task_node(e, FID_ir_eval_k101, WL_CONT, WL_IDX, 1);
      e.mem[_t_98 + 0] = _h_68;
      WL_CONT = term_tsk(FID_ir_eval_k101, _t_98);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_99 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_99 + 0] = _nodes_14;
      e.mem[_t_99 + 1] = _env_31;
      e.mem[_t_99 + 2] = _env_32;
      e.mem[_t_99 + 3] = _o_63;
      return term_tsk(FID_ir_eval, _t_99);
    }
    r0 = _nodes_14;
    r1 = _env_31;
    r2 = _env_32;
    r3 = _o_63;
    WL_JMP(FID_ir_eval);
  }}

  WL_CASE(FID_ir_eval_k101)
    WL_POPN(1);
    Term _h_70 = STK(0);
    Term _h_69 = r0;
    WL_OPEN
    Term _o_64[1];
    if (spin_ir_vge(e, _o_64, _h_70, _h_69) == 0) {
      return 0;
    }
    r0 = _o_64[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k103)
    WL_POPN(4);
    Term _nodes_15 = STK(0);
    Term _env_33 = STK(1);
    Term _env_34 = STK(2);
    u32 _o_65 = STK(3);
    Term _h_71 = r0;
    WL_OPEN
    if (term_aux(_h_71) == CID_ir_VB) {
      u32 _s_6 = term_loc(_h_71);
      if (_s_6 == 0) {
        term_sink(e, _nodes_15);
        term_sink(e, _env_33);
        term_sink(e, _env_34);
        r0 = term_pak(CID_ir_VB, 0);
        WL_RETN(1);
      } else {
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_eval_k105;
          WL_PUSHN(1);
        } else {
          u64 _t_101 = task_node(e, FID_ir_eval_k105, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_eval_k105, _t_101);
          WL_IDX = 0;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
          u64 _t_102 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
          e.mem[_t_102 + 0] = _nodes_15;
          e.mem[_t_102 + 1] = _env_33;
          e.mem[_t_102 + 2] = _env_34;
          e.mem[_t_102 + 3] = _o_65;
          return term_tsk(FID_ir_eval, _t_102);
        }
        r0 = _nodes_15;
        r1 = _env_33;
        r2 = _env_34;
        r3 = _o_65;
        WL_JMP(FID_ir_eval);
      }
    } else if (term_aux(_h_71) == CID_ir_VE) {
      term_sink(e, _nodes_15);
      term_sink(e, _env_33);
      term_sink(e, _env_34);
      Term _fb_33[1];
      u64 _sp_33 = ctr_take(e, _h_71, 1, _fb_33);
      Term _f_61 = _fb_33[0];
      u64 _nd_23 = _sp_33 >= HEAP_OFF ? _sp_33 : heap_alloc(e, 0);
      e.mem[_nd_23 + 0] = rfc_seal(e, _f_61);
      r0 = term_ctr(CID_ir_VE, _nd_23);
      WL_RETN(1);
    } else {
      term_sink(e, _nodes_15);
      term_sink(e, _env_33);
      term_sink(e, _env_34);
      term_sink(e, _h_71);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 1974);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_eval_k105)
    Term _h_72 = r0;
    WL_OPEN
    Term _o_66[1];
    if (spin_ir_booleanValue(e, _o_66, _h_72) == 0) {
      return 0;
    }
    r0 = _o_66[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k107)
    WL_POPN(4);
    Term _nodes_16 = STK(0);
    Term _env_35 = STK(1);
    Term _env_36 = STK(2);
    u32 _o_67 = STK(3);
    Term _h_73 = r0;
    WL_OPEN
    if (term_aux(_h_73) == CID_ir_VB) {
      u32 _s_7 = term_loc(_h_73);
      if (_s_7 == 1) {
        term_sink(e, _nodes_16);
        term_sink(e, _env_35);
        term_sink(e, _env_36);
        r0 = term_pak(CID_ir_VB, 1);
        WL_RETN(1);
      } else {
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_eval_k109;
          WL_PUSHN(1);
        } else {
          u64 _t_104 = task_node(e, FID_ir_eval_k109, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_eval_k109, _t_104);
          WL_IDX = 0;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
          u64 _t_105 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
          e.mem[_t_105 + 0] = _nodes_16;
          e.mem[_t_105 + 1] = _env_35;
          e.mem[_t_105 + 2] = _env_36;
          e.mem[_t_105 + 3] = _o_67;
          return term_tsk(FID_ir_eval, _t_105);
        }
        r0 = _nodes_16;
        r1 = _env_35;
        r2 = _env_36;
        r3 = _o_67;
        WL_JMP(FID_ir_eval);
      }
    } else if (term_aux(_h_73) == CID_ir_VE) {
      term_sink(e, _nodes_16);
      term_sink(e, _env_35);
      term_sink(e, _env_36);
      Term _fb_34[1];
      u64 _sp_34 = ctr_take(e, _h_73, 1, _fb_34);
      Term _f_62 = _fb_34[0];
      u64 _nd_24 = _sp_34 >= HEAP_OFF ? _sp_34 : heap_alloc(e, 0);
      e.mem[_nd_24 + 0] = rfc_seal(e, _f_62);
      r0 = term_ctr(CID_ir_VE, _nd_24);
      WL_RETN(1);
    } else {
      term_sink(e, _nodes_16);
      term_sink(e, _env_35);
      term_sink(e, _env_36);
      term_sink(e, _h_73);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 1925);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_eval_k109)
    Term _h_74 = r0;
    WL_OPEN
    Term _o_68[1];
    if (spin_ir_booleanValue(e, _o_68, _h_74) == 0) {
      return 0;
    }
    r0 = _o_68[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k111)
    Term _h_75 = r0;
    WL_OPEN
    Term _o_69[1];
    if (spin_ir_vnot(e, _o_69, _h_75) == 0) {
      return 0;
    }
    r0 = _o_69[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k113)
    WL_POPN(4);
    Term _nodes_17 = STK(0);
    Term _env_37 = STK(1);
    Term _env_38 = STK(2);
    u32 _o_70 = STK(3);
    Term _h_76 = r0;
    WL_OPEN
    if (term_aux(_h_76) == CID_ir_VB) {
      u32 _s_8 = term_loc(_h_76);
      if (_s_8 == 0) {
        term_sink(e, _nodes_17);
        term_sink(e, _env_37);
        term_sink(e, _env_38);
        r0 = term_pak(CID_ir_VB, 1);
        WL_RETN(1);
      } else {
        if (seq) {
          WL_ROOM(1);
          STK(0) = FID_ir_eval_k115;
          WL_PUSHN(1);
        } else {
          u64 _t_108 = task_node(e, FID_ir_eval_k115, WL_CONT, WL_IDX, 1);
          WL_CONT = term_tsk(FID_ir_eval_k115, _t_108);
          WL_IDX = 0;
        }
        if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
          u64 _t_109 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
          e.mem[_t_109 + 0] = _nodes_17;
          e.mem[_t_109 + 1] = _env_37;
          e.mem[_t_109 + 2] = _env_38;
          e.mem[_t_109 + 3] = _o_70;
          return term_tsk(FID_ir_eval, _t_109);
        }
        r0 = _nodes_17;
        r1 = _env_37;
        r2 = _env_38;
        r3 = _o_70;
        WL_JMP(FID_ir_eval);
      }
    } else if (term_aux(_h_76) == CID_ir_VE) {
      term_sink(e, _nodes_17);
      term_sink(e, _env_37);
      term_sink(e, _env_38);
      Term _fb_35[1];
      u64 _sp_35 = ctr_take(e, _h_76, 1, _fb_35);
      Term _f_63 = _fb_35[0];
      u64 _nd_25 = _sp_35 >= HEAP_OFF ? _sp_35 : heap_alloc(e, 0);
      e.mem[_nd_25 + 0] = rfc_seal(e, _f_63);
      r0 = term_ctr(CID_ir_VE, _nd_25);
      WL_RETN(1);
    } else {
      term_sink(e, _nodes_17);
      term_sink(e, _env_37);
      term_sink(e, _env_38);
      term_sink(e, _h_76);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 1827);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_ir_eval_k115)
    Term _h_77 = r0;
    WL_OPEN
    Term _o_71[1];
    if (spin_ir_booleanValue(e, _o_71, _h_77) == 0) {
      return 0;
    }
    r0 = _o_71[0];
    WL_RETN(1);
  }}

  WL_CASE(FID_ir_eval_k117)
    WL_POPN(5);
    Term _nodes_18 = STK(0);
    Term _env_39 = STK(1);
    Term _env_40 = STK(2);
    u32 _o_72 = STK(3);
    u32 _o_73 = STK(4);
    Term _h_78 = r0;
    WL_OPEN
    if (term_aux(_h_78) == CID_ir_VB) {
      u32 _b_0 = term_loc(_h_78);
      if (_b_0 == 1) {
        if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
          u64 _t_111 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
          e.mem[_t_111 + 0] = _nodes_18;
          e.mem[_t_111 + 1] = _env_39;
          e.mem[_t_111 + 2] = _env_40;
          e.mem[_t_111 + 3] = _o_72;
          return term_tsk(FID_ir_eval, _t_111);
        }
        r0 = _nodes_18;
        r1 = _env_39;
        r2 = _env_40;
        r3 = _o_72;
        WL_JMP(FID_ir_eval);
      } else {
        if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
          u64 _t_112 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
          e.mem[_t_112 + 0] = _nodes_18;
          e.mem[_t_112 + 1] = _env_39;
          e.mem[_t_112 + 2] = _env_40;
          e.mem[_t_112 + 3] = _o_73;
          return term_tsk(FID_ir_eval, _t_112);
        }
        r0 = _nodes_18;
        r1 = _env_39;
        r2 = _env_40;
        r3 = _o_73;
        WL_JMP(FID_ir_eval);
      }
    } else if (term_aux(_h_78) == CID_ir_VE) {
      term_sink(e, _nodes_18);
      term_sink(e, _env_39);
      term_sink(e, _env_40);
      Term _fb_36[1];
      u64 _sp_36 = ctr_take(e, _h_78, 1, _fb_36);
      Term _f_64 = _fb_36[0];
      u64 _nd_26 = _sp_36 >= HEAP_OFF ? _sp_36 : heap_alloc(e, 0);
      e.mem[_nd_26 + 0] = rfc_seal(e, _f_64);
      r0 = term_ctr(CID_ir_VE, _nd_26);
      WL_RETN(1);
    } else {
      term_sink(e, _nodes_18);
      term_sink(e, _env_39);
      term_sink(e, _env_40);
      term_sink(e, _h_78);
      r0 = term_ctr(CID_ir_VE, STAT_OFF + 1768);
      WL_RETN(1);
    }
  }}

  WL_CASE(FID_U32_show)
    u32 _a_0 = r0;
    WL_OPEN
    u32 _s_0 = U32_BIN(_a_0, ==, 0);
    if (_s_0 == 1) {
      r0 = term_ctr(CID_SCon, STAT_OFF + 2459);
      WL_RETN(1);
    } else {
      if (!DEVICE && !seq && fid_nofk(FID_U32_show_go)) {
        u64 _t_0 = task_node(e, FID_U32_show_go, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = 10ull;
        e.mem[_t_0 + 1] = _a_0;
        e.mem[_t_0 + 2] = term_pak(CID_SNil, 0);
        return term_tsk(FID_U32_show_go, _t_0);
      }
      r0 = 10ull;
      r1 = _a_0;
      r2 = term_pak(CID_SNil, 0);
      WL_JMP(FID_U32_show_go);
    }
  }}

  WL_CASE(FID_String_append)
    Term _a_0 = r0;
    Term _b_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_a_0) == CID_SNil) {
      r0 = _b_0;
      WL_RETN(1);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _a_0, 2, _fb_0);
      u32 _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      spare_free(e, cls_fit(2), _sp_0);
      if (seq) {
        WL_ROOM(2);
        STK(0) = _f_0;
        STK(1) = FID_String_append_k1;
        WL_PUSHN(2);
      } else {
        u64 _t_0 = task_node(e, FID_String_append_k1, WL_CONT, WL_IDX, 1);
        e.mem[_t_0 + 0] = _f_0;
        WL_CONT = term_tsk(FID_String_append_k1, _t_0);
        WL_IDX = 1;
      }
      r0 = _f_1;
      r1 = _b_0;
      _a_0 = r0;
      _b_0 = r1;
      WL_AGAIN(FID_String_append);
    }
    WL_SPUN
  }}

  WL_CASE(FID_String_append_k1)
    WL_POPN(1);
    u32 _f_2 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = rfc_seal(e, _f_2);
    e.mem[_nd_0 + 1] = rfc_seal(e, _h_0);
    r0 = term_ctr(CID_SCon, _nd_0);
    WL_RETN(1);
  }}

#if !DEVICE
  WL_CASE(FID_U32_read_go)
    Term _s_0 = r0;
    u32 _acc_0 = r1;
    WL_OPEN
    WL_SPIN
    if (term_aux(_s_0) == CID_SNil) {
      r0 = 1;
      r1 = _acc_0;
      WL_RETN(2);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _s_0, 2, _fb_0);
      u32 _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      Term _a_0 = U32_BIN(_acc_0, *, 10ull);
      Term _a_1 = U32_BIN(_f_0, -, 48ull);
      u32 _n_0 = U32_BIN(_a_0, +, _a_1);
      Term _a_2 = ((u32)(10ull) == 0 ? 0 : (u64)U32_QUO((u32)(_n_0), (u32)(10ull)));
      u32 _s_1 = U32_BIN(_a_2, ==, _acc_0);
      if (_s_1 == 1) {
        spare_free(e, cls_fit(2), _sp_0);
        r0 = _f_1;
        r1 = _n_0;
        _s_0 = r0;
        _acc_0 = r1;
        WL_AGAIN(FID_U32_read_go);
      } else {
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = 0;
        r1 = 0;
        WL_RETN(2);
      }
    }
    WL_SPUN
  }}
#endif

  WL_CASE(FID_calculate)
    u32 _id_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_ir_eval)) {
      u64 _t_0 = task_node(e, FID_ir_eval, WL_CONT, WL_IDX, 0);
      e.mem[_t_0 + 0] = term_ctr(CID_Con, STAT_OFF + 2467);
      e.mem[_t_0 + 1] = term_ctr(CID_Con, STAT_OFF + 2469);
      e.mem[_t_0 + 2] = term_pak(CID_Nil, 0);
      e.mem[_t_0 + 3] = _id_0;
      return term_tsk(FID_ir_eval, _t_0);
    }
    r0 = term_ctr(CID_Con, STAT_OFF + 2467);
    r1 = term_ctr(CID_Con, STAT_OFF + 2469);
    r2 = term_pak(CID_Nil, 0);
    r3 = _id_0;
    WL_JMP(FID_ir_eval);
  }}

#if !DEVICE
  WL_CASE(FID_U32_read)
    Term _s_0 = r0;
    WL_OPEN
    if (term_aux(_s_0) == CID_SNil) {
      r0 = 0;
      r1 = 0;
      WL_RETN(2);
    } else {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _s_0, 2, _fb_0);
      u32 _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      u64 _nd_0 = _sp_0 >= HEAP_OFF ? _sp_0 : heap_alloc(e, 1);
      e.mem[_nd_0 + 0] = rfc_seal(e, _f_0);
      e.mem[_nd_0 + 1] = rfc_seal(e, _f_1);
      if (!DEVICE && !seq && fid_nofk(FID_U32_read_go)) {
        u64 _t_0 = task_node(e, FID_U32_read_go, WL_CONT, WL_IDX, 0);
        e.mem[_t_0 + 0] = term_ctr(CID_SCon, _nd_0);
        e.mem[_t_0 + 1] = 0ull;
        return term_tsk(FID_U32_read_go, _t_0);
      }
      r0 = term_ctr(CID_SCon, _nd_0);
      r1 = 0ull;
      WL_JMP(FID_U32_read_go);
    }
  }}
#endif

#if !DEVICE
  WL_CASE(FID_IO_bind)
    Term _m_0 = r0;
    Term _f_0 = r1;
    Term _k_0 = r2;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = _f_0;
    e.mem[_nd_0 + 1] = _k_0;
    if (!DEVICE && !seq && fid_nofk(FID_Clo_apply)) {
      u64 _t_3 = task_node(e, FID_Clo_apply, WL_CONT, WL_IDX, 0);
      e.mem[_t_3 + 0] = _m_0;
      e.mem[_t_3 + 1] = term_clo(FID_IO_bind_c0, _nd_0);
      return term_tsk(FID_Clo_apply, _t_3);
    }
    r0 = _m_0;
    r1 = term_clo(FID_IO_bind_c0, _nd_0);
    WL_JMP(FID_Clo_apply);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_IO_bind_c0)
    Term _f_1 = r0;
    Term _k_1 = r1;
    Term _x_0 = r2;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _k_1;
      STK(1) = FID_IO_bind_k2;
      WL_PUSHN(2);
    } else {
      u64 _t_0 = task_node(e, FID_IO_bind_k2, WL_CONT, WL_IDX, 1);
      e.mem[_t_0 + 0] = _k_1;
      WL_CONT = term_tsk(FID_IO_bind_k2, _t_0);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_Clo_apply)) {
      u64 _t_1 = task_node(e, FID_Clo_apply, WL_CONT, WL_IDX, 0);
      e.mem[_t_1 + 0] = _f_1;
      e.mem[_t_1 + 1] = _x_0;
      return term_tsk(FID_Clo_apply, _t_1);
    }
    r0 = _f_1;
    r1 = _x_0;
    WL_JMP(FID_Clo_apply);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_IO_bind_k2)
    WL_POPN(1);
    Term _k_2 = STK(0);
    Term _h_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_Clo_apply)) {
      u64 _t_2 = task_node(e, FID_Clo_apply, WL_CONT, WL_IDX, 0);
      e.mem[_t_2 + 0] = _h_0;
      e.mem[_t_2 + 1] = _k_2;
      return term_tsk(FID_Clo_apply, _t_2);
    }
    r0 = _h_0;
    r1 = _k_2;
    WL_JMP(FID_Clo_apply);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main)
    WL_OPEN
    r0 = term_clo(FID_main_c0, 0);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c0)
    Term _x_0 = r0;
    WL_OPEN
    if (!DEVICE && !seq && fid_nofk(FID_IO_bind)) {
      u64 _t_31 = task_node(e, FID_IO_bind, WL_CONT, WL_IDX, 0);
      e.mem[_t_31 + 0] = term_clo(FID_metal_runtime_backend, 0);
      e.mem[_t_31 + 1] = term_clo(FID_main_c1, 0);
      e.mem[_t_31 + 2] = _x_0;
      return term_tsk(FID_IO_bind, _t_31);
    }
    r0 = term_clo(FID_metal_runtime_backend, 0);
    r1 = term_clo(FID_main_c1, 0);
    r2 = _x_0;
    WL_JMP(FID_IO_bind);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c1)
    Term _x_1 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 0);
    e.mem[_nd_0 + 0] = _x_1;
    r0 = term_clo(FID_main_c2, _nd_0);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c2)
    u32 _x_3 = r0;
    Term _x_2 = r1;
    WL_OPEN
    u64 _nd_1 = heap_alloc(e, 0);
    e.mem[_nd_1 + 0] = _x_3;
    if (!DEVICE && !seq && fid_nofk(FID_IO_bind)) {
      u64 _t_30 = task_node(e, FID_IO_bind, WL_CONT, WL_IDX, 0);
      e.mem[_t_30 + 0] = term_clo(FID_IO_args, 0);
      e.mem[_t_30 + 1] = term_clo(FID_main_c3, _nd_1);
      e.mem[_t_30 + 2] = _x_2;
      return term_tsk(FID_IO_bind, _t_30);
    }
    r0 = term_clo(FID_IO_args, 0);
    r1 = term_clo(FID_main_c3, _nd_1);
    r2 = _x_2;
    WL_JMP(FID_IO_bind);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c3)
    u32 _x_5 = r0;
    Term _x_4 = r1;
    WL_OPEN
    if (term_aux(_x_4) == CID_Con) {
      Term _fb_0[2];
      u64 _sp_0 = ctr_take(e, _x_4, 2, _fb_0);
      Term _f_0 = _fb_0[0];
      Term _f_1 = _fb_0[1];
      term_sink(e, _f_0);
      if (term_aux(_f_1) == CID_Con) {
        Term _fb_1[2];
        u64 _sp_1 = ctr_take(e, _f_1, 2, _fb_1);
        Term _f_2 = _fb_1[0];
        Term _f_3 = _fb_1[1];
        if (term_aux(_f_3) == CID_Nil) {
          spare_free(e, cls_fit(2), _sp_1);
          spare_free(e, cls_fit(2), _sp_0);
          if (seq) {
            WL_ROOM(2);
            STK(0) = _x_5;
            STK(1) = FID_main_k5;
            WL_PUSHN(2);
          } else {
            u64 _t_0 = task_node(e, FID_main_k5, WL_CONT, WL_IDX, 1);
            e.mem[_t_0 + 0] = _x_5;
            WL_CONT = term_tsk(FID_main_k5, _t_0);
            WL_IDX = 1;
          }
          if (!DEVICE && !seq && fid_nofk(FID_U32_read)) {
            u64 _t_1 = task_node(e, FID_U32_read, WL_CONT, WL_IDX, 0);
            e.mem[_t_1 + 0] = _f_2;
            return term_tsk(FID_U32_read, _t_1);
          }
          r0 = _f_2;
          WL_JMP(FID_U32_read);
        } else {
          term_sink(e, _f_2);
          term_sink(e, _f_3);
          spare_free(e, cls_fit(2), _sp_1);
          spare_free(e, cls_fit(2), _sp_0);
          r0 = term_clo(FID_main_c38, 0);
          WL_RETN(1);
        }
      } else {
        term_sink(e, _f_1);
        spare_free(e, cls_fit(2), _sp_0);
        r0 = term_clo(FID_main_c39, 0);
        WL_RETN(1);
      }
    } else {
      term_sink(e, _x_4);
      r0 = term_clo(FID_main_c40, 0);
      WL_RETN(1);
    }
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k5)
    WL_POPN(1);
    u32 _x_6 = STK(0);
    u32 _h_0 = r0;
    u32 _h_1 = r1;
    WL_OPEN
    if (_h_0 == 1) {
      if (seq) {
        WL_ROOM(2);
        STK(0) = _x_6;
        STK(1) = FID_main_k7;
        WL_PUSHN(2);
      } else {
        u64 _t_2 = task_node(e, FID_main_k7, WL_CONT, WL_IDX, 1);
        e.mem[_t_2 + 0] = _x_6;
        WL_CONT = term_tsk(FID_main_k7, _t_2);
        WL_IDX = 1;
      }
      if (!seq) {
        u64 _t_3 = task_node(e, FID_calculate, WL_CONT, WL_IDX, 0);
        e.mem[_t_3 + 0] = _h_1;
        return term_tsk(FID_calculate, _t_3);
      }
      r0 = _h_1;
      WL_JMP(FID_calculate);
    } else {
      r0 = term_clo(FID_main_c37, 0);
      WL_RETN(1);
    }
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k7)
    WL_POPN(1);
    u32 _x_7 = STK(0);
    Term _h_2 = r0;
    WL_OPEN
    if (term_aux(_h_2) == CID_ir_VI) {
      u32 _number_0 = term_loc(_h_2);
      u64 _nd_2 = heap_alloc(e, 1);
      e.mem[_nd_2 + 0] = _number_0;
      e.mem[_nd_2 + 1] = _x_7;
      r0 = term_clo(FID_main_c8, _nd_2);
      WL_RETN(1);
    } else {
      term_sink(e, _h_2);
      r0 = term_clo(FID_main_c36, 0);
      WL_RETN(1);
    }
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c8)
    u32 _number_1 = r0;
    u32 _x_9 = r1;
    Term _x_8 = r2;
    WL_OPEN
    u64 _nd_3 = heap_alloc(e, 1);
    e.mem[_nd_3 + 0] = _number_1;
    e.mem[_nd_3 + 1] = _x_9;
    if (!DEVICE && !seq && fid_nofk(FID_IO_bind)) {
      u64 _t_29 = task_node(e, FID_IO_bind, WL_CONT, WL_IDX, 0);
      e.mem[_t_29 + 0] = term_clo(FID_metal_runtime_events, 0);
      e.mem[_t_29 + 1] = term_clo(FID_main_c9, _nd_3);
      e.mem[_t_29 + 2] = _x_8;
      return term_tsk(FID_IO_bind, _t_29);
    }
    r0 = term_clo(FID_metal_runtime_events, 0);
    r1 = term_clo(FID_main_c9, _nd_3);
    r2 = _x_8;
    WL_JMP(FID_IO_bind);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c9)
    u32 _number_2 = r0;
    u32 _x_11 = r1;
    Term _x_10 = r2;
    WL_OPEN
    u64 _nd_4 = heap_alloc(e, 2);
    e.mem[_nd_4 + 0] = _number_2;
    e.mem[_nd_4 + 1] = _x_11;
    e.mem[_nd_4 + 2] = _x_10;
    r0 = term_clo(FID_main_c10, _nd_4);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c10)
    u32 _number_3 = r0;
    u32 _x_13 = r1;
    u32 _x_14 = r2;
    Term _x_12 = r3;
    WL_OPEN
    u64 _nd_5 = heap_alloc(e, 2);
    e.mem[_nd_5 + 0] = _number_3;
    e.mem[_nd_5 + 1] = _x_13;
    e.mem[_nd_5 + 2] = _x_14;
    if (!DEVICE && !seq && fid_nofk(FID_IO_bind)) {
      u64 _t_28 = task_node(e, FID_IO_bind, WL_CONT, WL_IDX, 0);
      e.mem[_t_28 + 0] = term_clo(FID_metal_runtime_heap_mib, 0);
      e.mem[_t_28 + 1] = term_clo(FID_main_c11, _nd_5);
      e.mem[_t_28 + 2] = _x_12;
      return term_tsk(FID_IO_bind, _t_28);
    }
    r0 = term_clo(FID_metal_runtime_heap_mib, 0);
    r1 = term_clo(FID_main_c11, _nd_5);
    r2 = _x_12;
    WL_JMP(FID_IO_bind);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c11)
    u32 _number_4 = r0;
    u32 _x_16 = r1;
    u32 _x_17 = r2;
    Term _x_15 = r3;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _number_4;
      STK(1) = _x_17;
      STK(2) = _x_15;
      STK(3) = FID_main_k13;
      WL_PUSHN(4);
    } else {
      u64 _t_4 = task_node(e, FID_main_k13, WL_CONT, WL_IDX, 1);
      e.mem[_t_4 + 0] = _number_4;
      e.mem[_t_4 + 1] = _x_17;
      e.mem[_t_4 + 2] = _x_15;
      WL_CONT = term_tsk(FID_main_k13, _t_4);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_U32_show)) {
      u64 _t_5 = task_node(e, FID_U32_show, WL_CONT, WL_IDX, 0);
      e.mem[_t_5 + 0] = _x_16;
      return term_tsk(FID_U32_show, _t_5);
    }
    r0 = _x_16;
    WL_JMP(FID_U32_show);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k13)
    WL_POPN(3);
    u32 _number_5 = STK(0);
    u32 _x_18 = STK(1);
    u32 _x_19 = STK(2);
    Term _h_3 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _number_5;
      STK(1) = _x_19;
      STK(2) = _h_3;
      STK(3) = FID_main_k15;
      WL_PUSHN(4);
    } else {
      u64 _t_6 = task_node(e, FID_main_k15, WL_CONT, WL_IDX, 1);
      e.mem[_t_6 + 0] = _number_5;
      e.mem[_t_6 + 1] = _x_19;
      e.mem[_t_6 + 2] = _h_3;
      WL_CONT = term_tsk(FID_main_k15, _t_6);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_U32_show)) {
      u64 _t_7 = task_node(e, FID_U32_show, WL_CONT, WL_IDX, 0);
      e.mem[_t_7 + 0] = _x_18;
      return term_tsk(FID_U32_show, _t_7);
    }
    r0 = _x_18;
    WL_JMP(FID_U32_show);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k15)
    WL_POPN(3);
    u32 _number_6 = STK(0);
    u32 _x_20 = STK(1);
    Term _h_5 = STK(2);
    Term _h_4 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _number_6;
      STK(1) = _h_5;
      STK(2) = _h_4;
      STK(3) = FID_main_k17;
      WL_PUSHN(4);
    } else {
      u64 _t_8 = task_node(e, FID_main_k17, WL_CONT, WL_IDX, 1);
      e.mem[_t_8 + 0] = _number_6;
      e.mem[_t_8 + 1] = _h_5;
      e.mem[_t_8 + 2] = _h_4;
      WL_CONT = term_tsk(FID_main_k17, _t_8);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_U32_show)) {
      u64 _t_9 = task_node(e, FID_U32_show, WL_CONT, WL_IDX, 0);
      e.mem[_t_9 + 0] = _x_20;
      return term_tsk(FID_U32_show, _t_9);
    }
    r0 = _x_20;
    WL_JMP(FID_U32_show);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k17)
    WL_POPN(3);
    u32 _number_7 = STK(0);
    Term _h_7 = STK(1);
    Term _h_8 = STK(2);
    Term _h_6 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _h_7;
      STK(1) = _h_8;
      STK(2) = _h_6;
      STK(3) = FID_main_k19;
      WL_PUSHN(4);
    } else {
      u64 _t_10 = task_node(e, FID_main_k19, WL_CONT, WL_IDX, 1);
      e.mem[_t_10 + 0] = _h_7;
      e.mem[_t_10 + 1] = _h_8;
      e.mem[_t_10 + 2] = _h_6;
      WL_CONT = term_tsk(FID_main_k19, _t_10);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_U32_show)) {
      u64 _t_11 = task_node(e, FID_U32_show, WL_CONT, WL_IDX, 0);
      e.mem[_t_11 + 0] = _number_7;
      return term_tsk(FID_U32_show, _t_11);
    }
    r0 = _number_7;
    WL_JMP(FID_U32_show);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k19)
    WL_POPN(3);
    Term _h_10 = STK(0);
    Term _h_11 = STK(1);
    Term _h_12 = STK(2);
    Term _h_9 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _h_10;
      STK(1) = _h_11;
      STK(2) = _h_12;
      STK(3) = FID_main_k21;
      WL_PUSHN(4);
    } else {
      u64 _t_12 = task_node(e, FID_main_k21, WL_CONT, WL_IDX, 1);
      e.mem[_t_12 + 0] = _h_10;
      e.mem[_t_12 + 1] = _h_11;
      e.mem[_t_12 + 2] = _h_12;
      WL_CONT = term_tsk(FID_main_k21, _t_12);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_13 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_13 + 0] = _h_9;
      e.mem[_t_13 + 1] = term_ctr(CID_SCon, STAT_OFF + 2471);
      return term_tsk(FID_String_append, _t_13);
    }
    r0 = _h_9;
    r1 = term_ctr(CID_SCon, STAT_OFF + 2471);
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k21)
    WL_POPN(3);
    Term _h_14 = STK(0);
    Term _h_15 = STK(1);
    Term _h_16 = STK(2);
    Term _h_13 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(4);
      STK(0) = _h_14;
      STK(1) = _h_15;
      STK(2) = _h_16;
      STK(3) = FID_main_k23;
      WL_PUSHN(4);
    } else {
      u64 _t_14 = task_node(e, FID_main_k23, WL_CONT, WL_IDX, 1);
      e.mem[_t_14 + 0] = _h_14;
      e.mem[_t_14 + 1] = _h_15;
      e.mem[_t_14 + 2] = _h_16;
      WL_CONT = term_tsk(FID_main_k23, _t_14);
      WL_IDX = 3;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_15 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_15 + 0] = term_ctr(CID_SCon, STAT_OFF + 2489);
      e.mem[_t_15 + 1] = _h_13;
      return term_tsk(FID_String_append, _t_15);
    }
    r0 = term_ctr(CID_SCon, STAT_OFF + 2489);
    r1 = _h_13;
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k23)
    WL_POPN(3);
    Term _h_18 = STK(0);
    Term _h_19 = STK(1);
    Term _h_20 = STK(2);
    Term _h_17 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(3);
      STK(0) = _h_18;
      STK(1) = _h_19;
      STK(2) = FID_main_k25;
      WL_PUSHN(3);
    } else {
      u64 _t_16 = task_node(e, FID_main_k25, WL_CONT, WL_IDX, 1);
      e.mem[_t_16 + 0] = _h_18;
      e.mem[_t_16 + 1] = _h_19;
      WL_CONT = term_tsk(FID_main_k25, _t_16);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_17 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_17 + 0] = _h_20;
      e.mem[_t_17 + 1] = _h_17;
      return term_tsk(FID_String_append, _t_17);
    }
    r0 = _h_20;
    r1 = _h_17;
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k25)
    WL_POPN(2);
    Term _h_22 = STK(0);
    Term _h_23 = STK(1);
    Term _h_21 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(3);
      STK(0) = _h_22;
      STK(1) = _h_23;
      STK(2) = FID_main_k27;
      WL_PUSHN(3);
    } else {
      u64 _t_18 = task_node(e, FID_main_k27, WL_CONT, WL_IDX, 1);
      e.mem[_t_18 + 0] = _h_22;
      e.mem[_t_18 + 1] = _h_23;
      WL_CONT = term_tsk(FID_main_k27, _t_18);
      WL_IDX = 2;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_19 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_19 + 0] = term_ctr(CID_SCon, STAT_OFF + 2501);
      e.mem[_t_19 + 1] = _h_21;
      return term_tsk(FID_String_append, _t_19);
    }
    r0 = term_ctr(CID_SCon, STAT_OFF + 2501);
    r1 = _h_21;
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k27)
    WL_POPN(2);
    Term _h_25 = STK(0);
    Term _h_26 = STK(1);
    Term _h_24 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_25;
      STK(1) = FID_main_k29;
      WL_PUSHN(2);
    } else {
      u64 _t_20 = task_node(e, FID_main_k29, WL_CONT, WL_IDX, 1);
      e.mem[_t_20 + 0] = _h_25;
      WL_CONT = term_tsk(FID_main_k29, _t_20);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_21 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_21 + 0] = _h_26;
      e.mem[_t_21 + 1] = _h_24;
      return term_tsk(FID_String_append, _t_21);
    }
    r0 = _h_26;
    r1 = _h_24;
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k29)
    WL_POPN(1);
    Term _h_28 = STK(0);
    Term _h_27 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(2);
      STK(0) = _h_28;
      STK(1) = FID_main_k31;
      WL_PUSHN(2);
    } else {
      u64 _t_22 = task_node(e, FID_main_k31, WL_CONT, WL_IDX, 1);
      e.mem[_t_22 + 0] = _h_28;
      WL_CONT = term_tsk(FID_main_k31, _t_22);
      WL_IDX = 1;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_23 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_23 + 0] = term_ctr(CID_SCon, STAT_OFF + 2517);
      e.mem[_t_23 + 1] = _h_27;
      return term_tsk(FID_String_append, _t_23);
    }
    r0 = term_ctr(CID_SCon, STAT_OFF + 2517);
    r1 = _h_27;
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k31)
    WL_POPN(1);
    Term _h_30 = STK(0);
    Term _h_29 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(1);
      STK(0) = FID_main_k33;
      WL_PUSHN(1);
    } else {
      u64 _t_24 = task_node(e, FID_main_k33, WL_CONT, WL_IDX, 1);
      WL_CONT = term_tsk(FID_main_k33, _t_24);
      WL_IDX = 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_25 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_25 + 0] = _h_30;
      e.mem[_t_25 + 1] = _h_29;
      return term_tsk(FID_String_append, _t_25);
    }
    r0 = _h_30;
    r1 = _h_29;
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k33)
    Term _h_31 = r0;
    WL_OPEN
    if (seq) {
      WL_ROOM(1);
      STK(0) = FID_main_k35;
      WL_PUSHN(1);
    } else {
      u64 _t_26 = task_node(e, FID_main_k35, WL_CONT, WL_IDX, 1);
      WL_CONT = term_tsk(FID_main_k35, _t_26);
      WL_IDX = 0;
    }
    if (!DEVICE && !seq && fid_nofk(FID_String_append)) {
      u64 _t_27 = task_node(e, FID_String_append, WL_CONT, WL_IDX, 0);
      e.mem[_t_27 + 0] = term_ctr(CID_SCon, STAT_OFF + 2535);
      e.mem[_t_27 + 1] = _h_31;
      return term_tsk(FID_String_append, _t_27);
    }
    r0 = term_ctr(CID_SCon, STAT_OFF + 2535);
    r1 = _h_31;
    WL_JMP(FID_String_append);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_k35)
    Term _h_32 = r0;
    WL_OPEN
    u64 _nd_6 = heap_alloc(e, 0);
    e.mem[_nd_6 + 0] = _h_32;
    r0 = term_clo(FID_IO_print, _nd_6);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c36)
    Term _x_21 = r0;
    WL_OPEN
    Term _o_0[1];
    if (spin_IO_die(e, _o_0, 2ull, term_ctr(CID_SCon, STAT_OFF + 2561), _x_21) == 0) {
      return 0;
    }
    r0 = _o_0[0];
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c37)
    Term _x_22 = r0;
    WL_OPEN
    Term _o_1[1];
    if (spin_IO_die(e, _o_1, 2ull, term_ctr(CID_SCon, STAT_OFF + 2603), _x_22) == 0) {
      return 0;
    }
    r0 = _o_1[0];
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c38)
    Term _x_23 = r0;
    WL_OPEN
    Term _o_2[1];
    if (spin_IO_die(e, _o_2, 2ull, term_ctr(CID_SCon, STAT_OFF + 2603), _x_23) == 0) {
      return 0;
    }
    r0 = _o_2[0];
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c39)
    Term _x_24 = r0;
    WL_OPEN
    Term _o_3[1];
    if (spin_IO_die(e, _o_3, 2ull, term_ctr(CID_SCon, STAT_OFF + 2603), _x_24) == 0) {
      return 0;
    }
    r0 = _o_3[0];
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_main_c40)
    Term _x_25 = r0;
    WL_OPEN
    Term _o_4[1];
    if (spin_IO_die(e, _o_4, 2ull, term_ctr(CID_SCon, STAT_OFF + 2603), _x_25) == 0) {
      return 0;
    }
    r0 = _o_4[0];
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_metal_runtime_backend)
    Term _k_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 0);
    e.mem[_nd_0 + 0] = _k_0;
    r0 = term_ctr(CID_metal_runtime_backend, _nd_0);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_IO_args)
    Term _k_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 0);
    e.mem[_nd_0 + 0] = _k_0;
    r0 = term_ctr(CID_IO_args, _nd_0);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_metal_runtime_events)
    Term _k_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 0);
    e.mem[_nd_0 + 0] = _k_0;
    r0 = term_ctr(CID_metal_runtime_events, _nd_0);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_metal_runtime_heap_mib)
    Term _k_0 = r0;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 0);
    e.mem[_nd_0 + 0] = _k_0;
    r0 = term_ctr(CID_metal_runtime_heap_mib, _nd_0);
    WL_RETN(1);
  }}
#endif

#if !DEVICE
  WL_CASE(FID_IO_print)
    Term _text_0 = r0;
    Term _k_0 = r1;
    WL_OPEN
    u64 _nd_0 = heap_alloc(e, 1);
    e.mem[_nd_0 + 0] = _text_0;
    e.mem[_nd_0 + 1] = _k_0;
    r0 = term_ctr(CID_IO_print, _nd_0);
    WL_RETN(1);
  }}
#endif

  WL_CASE(FID_ENTER)
    Term t = r0;
    WL_OPEN
    u32 f   = term_aux(t);
    u64 a   = term_loc(t);
    u32 war = fid_arity(f);
    WL_FRAME(t)
    seq |= fid_nofk(f) << 1;
    u32 rw = fid_resw(f);
    if (rw) {
      WL_LOAD(a + war - rw, rw)
      WL_ARGS(a, war - rw)
    } else {
      WL_LOAD(a, war)
    }
    heap_free(e, cls_fit(war + 2), a);
    WL_DYN(f);
  }}

  WL_CASE(FID_IO_emit)
    Term x = r0;
    WL_OPEN
    u64 l = heap_alloc(e, 0);
    e.mem[l] = x;
    r0 = term_ctr(CID_Emit, l);
    WL_RETN(1);
  }}

  WL_CASE(FID_Clo_apply)
    Term fun = r0;
    Term arg = r1;
    WL_OPEN
    u32 f    = term_aux(fun);
    u32 war  = fid_arity(f) - 1;
    u64 a    = term_loc(fun);
    WL_LOAD(a, war)
    spare_free(e, cls_fit(war), a);
    WL_LAST(arg)
    WL_DYN(f);
  }}

  WL_CASE(FID_EXIT)
    u32  n = rn;
    Term rv[WL_RESW];
    WL_SAVE(rv)
    WL_OPEN
    if (err_seen(e.mem)) {
      return 0;
    }
    WL_POPN(2);
    Term cont = STK(0);
    u32  idx  = (u32)STK(1);
    u32  wf   = term_aux(cont);
    if (cont != TERM_HOLE && fid_resw(wf)) {
      u64 wa = term_loc(cont);
      u32 wn = fid_arity(wf);
      WL_FRAME(cont)
      seq = (seq & 1) | fid_nofk(wf) << 1;
      WL_ARGS(wa, wn - n)
      heap_free(e, cls_fit(wn + 2), wa);
      WL_TAKE(rv)
      WL_DYN(wf);
    }
    return task_deliver(e.mem, cont, idx, rv, n);
  }}

#if DEVICE
  default: {
    err_post(e.mem, ERR_FIDS);
    return 0;
  }
  }
  }
}
#endif

// Monk
// ====

// One turn on a ring: its head task below put0 runs (a growing lane skips a
// fork-free one, unless on the host a forking frame follows it).
INLINE u32 monk_step(Env e, DEV Term* stk, u32 rg, u32 put0, u32 base, u32 stride,
  TG u32* cur) {
  DEV u64* H   = e.mem;
  bool     seq = stride == 0;
  DEV u32* get = ring_get(H, rg);
  if (*get == put0) {
    return 0;
  }
  DEV u32* lo = (DEV u32*)ring_slot(H, rg, *get);
  u32      hi = a32_load_acq(lo + 1);
  Term     t  = (((u64)hi << 32) | a32_load(lo)) & ~RFC_BIT;
  if ((hi >> 31) != ring_lap(*get) || (!seq && fid_nofk(term_aux(t)))) {
    return 0;
  }
  a32_store(get, *get + 1);
  u32 spin = 0;
  for (;;) {
    Term r = work_loop(e, stk, t, seq);
    if (r == 0) {
      return 2;
    }
    if ((u32)H[task_tail(r) + 1] == 0) {
      bool nofk = fid_nofk(term_aux(r));
      Term k    = H[task_tail(r)];
      if (err_spun(H, &spin)) {
        return 2;
      }
      if (stride != 0 && nofk && (DEVICE || k == TERM_HOLE
        || !fid_resw(term_aux(k)) || fid_nofk(term_aux(k)))) {
        ring_push(H, ring_pick(base, stride, cur), r);
        return 2;
      }
      t      = r;
      seq    = false;
      stride = nofk ? stride : 0;
      continue;
    }
    task_deal(H, r, base, stride, cur);
    return 1;
  }
}

// Dev
// ===

// One kernel: pass 0 grows the frontier until its group is full or nothing
// grew, pass 1 runs the c-th deal on lane c, pass 3 is pass 1 on the lane's
// own ring after a small grow, and pass 2 slides each bank's [top, wr) onto
// rd in one group. TG_HOLD words of threadgroup memory keep one group per
// Apple core.

#if DEVICE

INLINE void dev_cut(Env e) {
  if (err_seen(e.mem)) {
    return;
  }
  for (u32 c = 0; c < NCLS_ALL; c += 1) {
    u64 gen = (u64)KEEP(c) << c;
    while (ALC_LEN(e, c) >= gen) {
      u64 head = ALC_AT(e, c);
      u64 tail = head;
      for (u32 i = 1; i < KEEP(c); i += 1) {
        tail = e.mem[tail];
      }
      ALC_AT(e, c)    = e.mem[tail];
      ALC_LEN(e, c)  -= gen;
      e.mem[tail]     = 0;
      bank_push(e.mem, c, head);
    }
  }
}

INLINE void bank_pack(DEV u64* H, u32 lane) {
  for (u32 c = 0; c < NCLS_ALL; c += 1) {
    DEV Bank* b  = bank_at(H, c);
    u32       rd = b->rd;
    u32       n  = b->wr - b->top;
    for (u32 i = 0; i < n; i += CUBE_T) {
      Term v = i + lane < n ? H[b->off + b->top + i + lane] : 0;
      BAR();
      if (i + lane < n) {
        H[b->off + rd + i + lane] = v;
      }
    }
    BAR();
    if (lane == 0) {
      b->rd = b->wr = b->top = rd + n;
    }
  }
}

#ifdef __METAL_VERSION__
kernel void bend_dev(DEV u64* H [[buffer(0)]], constant u32& pass [[buffer(1)]],
  TG u32* vote [[threadgroup(0)]],
  u32 grids [[threadgroups_per_grid]],
  u32 row [[threadgroup_position_in_grid]],
  u32 lane [[thread_position_in_threadgroup]]) {
#else
extern "C" __global__ void bend_dev(DEV u64* H, u32 pass, u32 at) {
  __shared__ u32 vote[3];
  u32 grids = gridDim.x;
  u32 row   = blockIdx.x;
  u32 lane  = threadIdx.x;
  if (!turn_groups(dev_front, at)) {
    return;
  }
#endif
  if (pass == 2) {
    bank_pack(H, lane);
    if (lane == 0) {
      turn_set(H, root_done(H) || err_seen(H) ? 0
        : turn_take(H));
    }
    return;
  }
  u32 stride = grids == 1 ? CUBE_G : 1;
  u32 me     = row * CUBE_T + stride * lane;
  u32 rg     = pass == 1 ? ring_flip(me) : me;
  Env e      = { H, H + ALC_OFF + me };
  DEV Term* stk = (DEV Term*)(H + STAK_OFF + me);
  if (lane == 0) {
    for (u32 i = 0; i < 3; i += 1) {
      a32_store(vote + i, 0);
    }
  }
  BAR();
  u32 put0      = a32_load(pass == 3 ? ring_held(H, rg) : ring_put(H, rg));
  u32 seen_has  = 0;
  u32 seen_grew = 0;
  for (;;) {
    if (pass) {
      if (*ring_get(H, rg) == put0 || err_seen(H)) {
        break;
      }
    } else {
      put0 = a32_load(ring_put(H, rg));
      u32 has = put0 != a32_load(ring_get(H, rg));
      if (lane == 0 && (err_seen(H) || root_done(H))) {
        has = CUBE_T;
      }
      a32_add(vote + 2, has);
      BAR();
      has = a32_load(vote + 2);
      if (has - seen_has >= CUBE_T) {
        break;
      }
      seen_has = has;
    }
    u32 ran = monk_step(e, stk, rg, put0, row * CUBE_T, pass ? 0 : stride,
      vote);
    if (!pass) {
      if (ran == 1) {
        a32_add(vote + 1, 1);
      }
      BARD();
      u32 grew = a32_load(vote + 1);
      if (grew == seen_grew) {
        break;
      }
      seen_grew = grew;
    }
  }
  if (pass == 0) {
    a32_store(ring_held(H, rg), a32_load(ring_put(H, rg)));
  }
  dev_cut(e);
}

#endif

// Window
// ======

// Linux's window fill (the Mac's is window_msl): an Image is a quadtree over
// 2^k x 2^k (Qua splits tl, tr, bl, br; Pix is 0xRRGGBB).
#if __linux__ || BEND_RTC

INLINE u32 window_pix(DEV u64* H, Term t, u32 k, u32 x, u32 y) {
  for (u32 i = k; term_tag(t) == TAG_CTR;) {
    u32 j = 0;
    if (i > 0) {
      i -= 1;
      j = ((y >> i) & 1) * 2 + ((x >> i) & 1);
    }
    t = H[term_peek(H, t) + j];
  }
  return (u32)term_loc(t) & 0xFFFFFF;
}

#ifdef BEND_RTC
extern "C" __global__ void window_dev(DEV u64* H, Term root, u32 w, u32 h,
  u32 k, u32* out) {
  u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  u32 y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    out[y * w + x] = window_pix(H, root, k, x, y);
  }
}
#endif

#endif

#if !DEVICE

// Row
// ===

static u64 io_tick(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// A row runs what precedes its forks on one thread; with fewer rows than
// threads, a grow past 1 ms deals its forks over the rows for idle threads.
static u32 row_grow(Env e, DEV Term* stk, u32 base, u32 stride, u32 want,
  u32 rows) {
  u64* H = e.mem;
  u32 cur = 0;
  u64 end = rows < pool_size ? io_tick() + 1000000 : ~0ull;
  for (;;) {
    u32 put0[CUBE_T];
    u32 has = 0;
    for (u32 i = 0; i < CUBE_T; i += 1) {
      put0[i] = *ring_put(H, base + i * stride);
      has += put0[i] != *ring_get(H, base + i * stride);
    }
    if (root_done(H) || has >= want) {
      return cur;
    }
    bool slow = io_tick() > end;
    u32 grew = 0;
    u32 ran  = 0;
    for (u32 i = 0; i < CUBE_T && ran != 2; i += 1) {
      ran   = monk_step(e, stk, base + i * stride, put0[i], slow ? 0 : base,
        slow ? CUBE_G : stride, slow ? a32_at(H, H_CURSOR) : &cur);
      grew += ran == 1;
    }
    if (grew == 0 || slow) {
      return cur;
    }
  }
}

// Pool
// ====

static void* pool_try(void* at, u64 bytes) {
  return mmap(at, bytes, PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
}

static void* pool_mmap(u64 bytes) {
  void* p = pool_try(NULL, bytes);
  if (p == MAP_FAILED) {
    err_fail("reservation failed");
  }
  return p;
}

static Term* pool_stack(void) {
  u64   len = 1ull << 31;
  u64   alt = 4 * SIGSTKSZ;
  char* p   = pool_mmap(len + 16384 + alt);
  if (mprotect(p + len, 16384, PROT_NONE)) {
    err_fail("stack guard failed");
  }
  stack_t ss = { .ss_sp = p + len + 16384, .ss_size = alt };
  sigaltstack(&ss, NULL);
  struct sigaction sa = { .sa_handler = err_trap,
    .sa_flags = SA_ONSTACK | SA_NODEFER };
  sigaction(SIGSEGV, &sa, NULL);
  sigaction(SIGBUS, &sa, NULL);
  return (Term*)p;
}

// A wait yields a while before it sleeps, so a short turn pays no wake.

#define POOL_WAIT(c, cv) \
  for (u32 s = 0; s < 128 && (c); s += 1) { \
    sched_yield(); \
  } \
  pthread_mutex_lock(&pool_lock); \
  while (c) { \
    pthread_cond_wait(&cv, &pool_lock); \
  } \
  pthread_mutex_unlock(&pool_lock);

static u32 pool_step(u32 rows) {
  u32 per = rows * CUBE_T / (32 * pool_size);
  return CUBE_T >> (31 - clz(per < LINE ? per | 1 : LINE));
}

static u32 pool_rows(Env e, DEV Term* stk) {
  u32 n = 0;
  for (;;) {
    u32 c    = a32_add(&pool_row, 1);
    u32 r    = c & 32767;
    u32 rows = c >> 15 & 255;
    u32 step = c >> 23 & 1 ? 1 : pool_step(rows);
    if (r >= rows * step) {
      if (n != 0 && a32_sub_rel(&pool_done, n) == n) {
        pthread_mutex_lock(&pool_lock);
        pthread_cond_signal(&pool_join);
        pthread_mutex_unlock(&pool_lock);
      }
      return c >> 24;
    }
    a32_acq(&pool_row);
    if (c >> 23 & 1) {
      row_grow(e, stk, r * CUBE_T, 1, CUBE_T, rows);
    } else {
      u32 row = r / step * CUBE_T;
      for (u32 rg = row + r % step; rg < row + CUBE_T; rg += step) {
        u32 put0 = a32_load(ring_put(e.mem, rg));
        while (*ring_get(e.mem, rg) != put0) {
          monk_step(e, stk, rg, put0, rg, 0, NULL);
        }
      }
    }
    n += 1;
  }
}

static void* pool_work(void* arg) {
  Term* stk  = pool_stack();
  u32   seen = 0;
  for (;;) {
    POOL_WAIT(a32_load(&pool_row) >> 24 == seen, pool_wake)
    seen = pool_rows((Env){ CORPUS, ALC[(uintptr_t)arg] }, stk);
  }
}

OUTLINE void pool_open(void) {
  static bool up;
  if (up) {
    return;
  }
  up = true;
  for (u32 w = 1; w < pool_size; w += 1) {
    pthread_t tid;
    if (pthread_create(&tid, NULL, pool_work, (void*)(uintptr_t)w)) {
      err_fail("pthread_create");
    }
  }
}

static int cpu_read(const char* path, long* a, long* b) {
  FILE* f = fopen(path, "r");
  if (f == NULL) {
    return 0;
  }
  int n = fscanf(f, "%ld %ld", a, b);
  fclose(f);
  return n;
}

static long cpu_count(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef __linux__
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) == 0) {
    n = CPU_COUNT(&set);
  }
  long q = 0;
  long p = 0;
  if (cpu_read("/sys/fs/cgroup/cpu.max", &q, &p) != 2) {
    cpu_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", &q, &p);
    cpu_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us", &p, &p);
  }
  if (q > 0 && p > 0 && (q + p - 1) / p < n) {
    n = (q + p - 1) / p;
  }
#endif
  return n;
}

OUTLINE void pool_turn(bool grow, u32 rows) {
  static u32 turn;
  u32 n = rows < CUBE_G ? rows : CUBE_G;
  turn += 1;
  a32_store(&pool_done, grow ? n : n * pool_step(n));
  pthread_mutex_lock(&pool_lock);
  a32_store_rel(&pool_row, turn << 24 | grow << 23 | n << 15);
  pthread_cond_broadcast(&pool_wake);
  pthread_mutex_unlock(&pool_lock);
  pool_rows((Env){ CORPUS, ALC[0] }, io_stk);
  POOL_WAIT(a32_load_acq(&pool_done) != 0, pool_join)
}

// Gpu
// ===

// gpu_make compiles the device program into <binary>.gpu: Metal's binary
// archive, or CUDA's cubin behind a hash of the text. CUDA sizes the bag by
// L2 (a group per 64 KB, 16..128) and runs one stream, which starts faster.

static const char* gpu_path(void) {
  static char path[4096];
  u32 n = sizeof path - 8;
#ifdef __APPLE__
  _NSGetExecutablePath(path, &n);
#else
  path[readlink("/proc/self/exe", path, n)] = 0;
#endif
  return strcat(path, ".gpu");
}

static void gpu_note(const char* path) {
  fprintf(stderr, "bend: compiling the GPU program (%s is missing or"
    " stale)\n", path);
}

#if !BEND_CUDA
#define gpu_map pool_mmap
#endif

#if BEND_METAL || BEND_CUDA

static void gpu_kernel(u32 pass, u32 at);

static void gpu_run(void) {
  for (u32 i = 0; i < 5; i += 1) {
    gpu_kernel((u32[]){ 0, 0, 1, 3, 2 }[i], i);
  }
}

#endif

#if BEND_METAL

static void gpu_fail(NSError* err) {
  err_fail([[err localizedDescription] UTF8String]);
}

static const char* gpu_probe(void) {
  gpu_dev = MTLCreateSystemDefaultDevice();
  return gpu_dev == nil ? "this binary found no usable Metal GPU" : NULL;
}

static MTLComputePipelineDescriptor* gpu_desc(void) {
  NSError* err = nil;
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.mathMode = MTLMathModeSafe;
  opts.preprocessorMacros = @{ @"CUBE_LOG": @(CUBE_LOG) };
  id<MTLLibrary> lib = [gpu_dev newLibraryWithSource:@(BEND_SRC) options:opts
    error:&err];
  if (!lib) {
    gpu_fail(err);
  }
  MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
  d.computeFunction = [lib newFunctionWithName:@"bend_dev"];
  return d;
}

static bool gpu_make(const char* path) {
  NSError* err = nil;
  id<MTLBinaryArchive> ar = [gpu_dev
    newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
  if (![ar addComputePipelineFunctionsWithDescriptor:gpu_desc() error:&err]) {
    gpu_fail(err);
  }
  return [ar serializeToURL:[NSURL fileURLWithPath:@(path)] error:&err];
}

static id<MTLComputePipelineState> gpu_pipe(MTLComputePipelineDescriptor* d,
  id<MTLBinaryArchive> ar) {
  NSError* err = nil;
  d.binaryArchives = ar ? @[ar] : @[];
  id<MTLComputePipelineState> pso = [gpu_dev
    newComputePipelineStateWithDescriptor:d
    options:ar ? MTLPipelineOptionFailOnBinaryArchiveMiss : 0 reflection:nil
    error:&err];
  if (!pso && !ar) {
    gpu_fail(err);
  }
  return pso;
}

static u64 gpu_span(void) {
  u64 span = [gpu_dev recommendedMaxWorkingSetSize];
  u64 most = [gpu_dev maxBufferLength];
  span = span < most ? span : most;
  return span < (2ull << 30) ? span : 2ull << 30;
}

static void gpu_load(u64 bytes) {
  gpu_buf = [gpu_dev newBufferWithBytesNoCopy:CORPUS length:bytes
    options:MTLResourceStorageModeShared
      | MTLResourceHazardTrackingModeUntracked deallocator:nil];
  if (!gpu_buf) {
    u64  most = [gpu_dev maxBufferLength];
    char msg[96];
    snprintf(msg, sizeof msg, "--gpu %lluMB is over the device's %lluMB",
      (unsigned long long)(bytes >> 20), (unsigned long long)(most >> 20));
    err_fail(bytes > most ? msg : "the GPU span is more than the device has");
  }
  @autoreleasepool {
    gpu_que = [gpu_dev newCommandQueue];
    const char* path = gpu_path();
    MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
    ad.url = [NSURL fileURLWithPath:@(path)];
    MTLComputePipelineDescriptor* d = gpu_desc();
    id<MTLBinaryArchive> ar = [gpu_dev newBinaryArchiveWithDescriptor:ad
      error:nil];
    gpu_pso = ar ? gpu_pipe(d, ar) : nil;
    if (!gpu_pso) {
      gpu_note(path);
      gpu_pso = gpu_pipe(d, nil);
    }
  }
}

static void gpu_kernel(u32 pass, u32 at) {
  [gpu_enc setBytes:&pass length:sizeof pass atIndex:1];
  [gpu_enc dispatchThreadgroupsWithIndirectBuffer:gpu_buf
    indirectBufferOffset:H_GROUPS * 8 + at * 12
    threadsPerThreadgroup:MTLSizeMake(CUBE_T, 1, 1)];
  [gpu_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

// GPU_DEEP rounds in flight, chained by one event, until the frontier empties.

#define GPU_DEEP 3

static id<MTLEvent> gpu_ev;
static u64          gpu_ev_at;

static void gpu_pass(u64* H) {
  id<MTLCommandBuffer> cb[GPU_DEEP] = { nil };
  gpu_ev = gpu_ev ? gpu_ev : [gpu_dev newEvent];
  for (u32 i = 0, left = GPU_DEEP; left != 0; i += 1) {
    @autoreleasepool {
      id<MTLCommandBuffer> c = cb[i % GPU_DEEP];
      [c waitUntilCompleted];
      if ([c error]) {
        gpu_fail([c error]);
      }
      left -= left < GPU_DEEP || ((volatile u64*)H)[H_FRONT] == 0;
      cb[i % GPU_DEEP] = c = left == GPU_DEEP ? [gpu_que commandBuffer] : nil;
      if (c) {
        [c encodeWaitForEvent:gpu_ev value:gpu_ev_at];
        gpu_enc = [c computeCommandEncoder];
        [gpu_enc setComputePipelineState:gpu_pso];
        [gpu_enc setBuffer:gpu_buf offset:0 atIndex:0];
        [gpu_enc setThreadgroupMemoryLength:TG_HOLD * 8 atIndex:0];
        gpu_run();
        [gpu_enc endEncoding];
        [c encodeSignalEvent:gpu_ev value:++gpu_ev_at];
        [c commit];
      }
    }
  }
}

#elif BEND_CUDA

static u64 gpu_hash(void) {
  u64 key = 14695981039346656037ull ^ CUBE_LOG;
  for (const char* p = BEND_SRC; *p != 0; p += 1) {
    key = (key ^ (u8)*p) * 1099511628211ull;
  }
  return key;
}

#define GPU_CHECK(fn, ...) do { \
  if ((result = fn(__VA_ARGS__)) != CUDA_SUCCESS) { why = #fn; goto fail; } \
} while (0)

static const char* gpu_probe(void) {
  static char error[192];
  const char* why;
  CUresult result;
  int managed = 0;
  CUcontext ctx;
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 0);
  GPU_CHECK(cuInit, 0);
  GPU_CHECK(cuDeviceGet, &gpu_dev, 0);
  GPU_CHECK(cuDeviceGetAttribute, &managed,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, gpu_dev);
  if (managed == 0) {
    return "CUDA device lacks concurrent managed access (WSL2 lacks it)";
  }
  int l2 = 1 << 23;
  cuDeviceGetAttribute(&l2, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, gpu_dev);
  int units = l2 >> 16;
  CUBE_LOG  = 31 - clz(units < 16 ? 16 : units > 128 ? 128 : units);
  GPU_CHECK(cuDevicePrimaryCtxRetain, &ctx, gpu_dev);
  result = cuCtxSetCurrent(ctx);
  if (result != CUDA_SUCCESS) {
    cuDevicePrimaryCtxRelease(gpu_dev);
    why = "cuCtxSetCurrent";
    goto fail;
  }
  return NULL;
fail:;
  const char* name = NULL;
  if (cuGetErrorName(result, &name) != CUDA_SUCCESS || name == NULL) {
    name = "unknown CUDA error";
  }
  snprintf(error, sizeof error, "%s failed: %s", why, name);
  return error;
}

#undef GPU_CHECK

static u64* gpu_map(u64 bytes) {
  CUdeviceptr p = 0;
  if (cuMemAllocManaged(&p, bytes, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) {
    err_fail("corpus reservation failed");
  }
#if CUDA_VERSION >= 13000
  cuMemAdvise(p, bytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
    (CUmemLocation){ CU_MEM_LOCATION_TYPE_DEVICE, gpu_dev });
#else
  cuMemAdvise(p, bytes, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, gpu_dev);
#endif
  cuMemsetD8(p, 0, STAK_OFF * 8);
  cuCtxSynchronize();
  return (u64*)(uintptr_t)p;
}

static bool gpu_make(const char* path) {
  int cc[2] = {0, 0};
  cuDeviceGetAttribute(cc,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, gpu_dev);
  cuDeviceGetAttribute(cc + 1,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, gpu_dev);
  char arch[40];
  char bag[24];
  snprintf(arch, sizeof arch, "--gpu-architecture=sm_%d%d", cc[0], cc[1]);
  snprintf(bag, sizeof bag, "-DCUBE_LOG=%u", CUBE_LOG);
  const char* opts[] = { arch, bag, "--fmad=false", "-default-device" };
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, BEND_SRC, "bend.cu", 0, NULL, NULL)
    != NVRTC_SUCCESS) {
    err_fail("cannot compile the CUDA library");
  }
  if (nvrtcCompileProgram(prog, 4, opts) != NVRTC_SUCCESS) {
    size_t n = 0;
    nvrtcGetProgramLogSize(prog, &n);
    char* log = calloc(n + 1, 1);
    if (log != NULL && nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS) {
      fprintf(stderr, "%s\n", log);
    }
    err_fail("cannot compile the CUDA library");
  }
  size_t len = 0;
  nvrtcGetCUBINSize(prog, &len);
  char* bin = malloc(len);
  if (bin == NULL || nvrtcGetCUBIN(prog, bin) != NVRTC_SUCCESS) {
    err_fail("cannot load the CUDA library");
  }
  nvrtcDestroyProgram(&prog);
  u64   key = gpu_hash();
  FILE* out = fopen(path, "wb");
  bool  ok  = out != NULL && fwrite(&key, 8, 1, out) == 1
    && fwrite(bin, 1, len, out) == len && fclose(out) == 0;
  if (cuModuleLoadData(&gpu_lib, bin) != CUDA_SUCCESS) {
    err_fail("cannot load the CUDA library");
  }
  free(bin);
  return ok;
}

static u64 gpu_span(void) {
  size_t span = 0;
  cuDeviceTotalMem(&span, gpu_dev);
  return span;
}

static void gpu_load(u64 bytes) {
  const char* path = gpu_path();
  int         fd   = open(path, O_RDONLY);
  struct stat st   = { 0 };
  u64         key  = 0;
  char*       bin  = fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 8 ? NULL
    : mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (bin != NULL && bin != MAP_FAILED) {
    memcpy(&key, bin, 8);
  }
  if (key != gpu_hash()
    || cuModuleLoadData(&gpu_lib, bin + 8) != CUDA_SUCCESS) {
    gpu_note(path);
    gpu_make(path);
  }
  if (cuModuleGetFunction(&gpu_pso, gpu_lib, "bend_dev") != CUDA_SUCCESS
    || cuModuleGetGlobal(&gpu_front, NULL, gpu_lib, "dev_front")
    != CUDA_SUCCESS) {
    err_fail("cannot load the GPU program");
  }
}

static void gpu_kernel(u32 pass, u32 at) {
  void* args[] = { &CORPUS, &pass, &at };
  if (cuLaunchKernel(gpu_pso, TURN_ROWS(at), 1, 1, CUBE_T, 1, 1, 0, NULL,
    args, NULL) != CUDA_SUCCESS) {
    err_fail("device launch failed");
  }
}

// A pass with no frontier returns at once.
static void gpu_pass(u64* H) {
  if (cuMemcpyHtoD(gpu_front, &H[H_FRONT], 8) != CUDA_SUCCESS) {
    err_fail("device fault");
  }
  for (u32 n = 1; H[H_FRONT] != 0; n = n < 32 ? 2 * n : 64) {
    for (u32 i = 0; i < n; i += 1) {
      gpu_run();
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) {
      err_fail("device fault");
    }
  }
}

#else

#define gpu_probe() "this binary has no CUDA or Metal GPU support"
#define gpu_make(p) true
#define gpu_span()  0
#define gpu_load(b)
#define gpu_pass(H)

#endif

// Cube
// ====

// The host's column grows only to a LINE of tasks a thread, as every task a
// grow starts is live at once; pool_step splits the few rows instead. A turn
// visits only the rows below its bound, which hold every task: a drain deals
// task g to ring_flip(g), at row g at most, and a grow's cur tasks land in
// rows 0 to cur - 1. A slow grow deals task g to row g % CUBE_T.

static void ring_rewind(u64* H, u32 rows) {
  Term keep[RING_LEN];
  for (u32 r = 0; r < (rows < CUBE_G ? rows : CUBE_G) * CUBE_T; r += 1) {
    u32 g = *ring_get(H, r);
    u32 n = *ring_put(H, r) - g;
    if (g == 0) {
      continue;
    }
    for (u32 i = 0; i < n; i += 1) {
      keep[i] = *ring_slot(H, r, g + i) & ~RFC_BIT;
    }
    for (u32 i = n; i < g + n && i < RING_LEN; i += 1) {
      ((u32*)ring_slot(H, r, i))[1] = 0;
    }
    *ring_get(H, r) = *ring_put(H, r) = 0;
    for (u32 i = 0; i < n; i += 1) {
      ring_push(H, r, keep[i]);
    }
  }
}

static void cube_run(u64* H, bool gpu) {
  u32 f = turn_take(H);
  if (gpu) {
    turn_set(H, f);
  }
  while (!root_done(H)) {
    if (f == 0) {
      err_fail("frontier drained without a result");
    }
    if (gpu) {
      gpu_pass(H);
      f = (u32)H[H_FRONT];
    } else {
      u32 rows = f;
      u32 want = (pool_size - 1) / (CUBE_T / LINE) + 1;
      if (f < want) {
        u32 cur = row_grow((Env){ H, ALC[0] }, io_stk, 0, CUBE_G, want, f);
        rows = cur > f ? cur : f;
      }
      for (u32 g = f; g != 0 && f < CUBE;
        g = turn_take(H)) {
        rows = g > rows ? g : rows;
        pool_turn(true, rows);
      }
      pool_turn(false, rows);
      f = turn_take(H);
      ring_rewind(H, f > rows ? f : rows);
    }
    u32 ec = a32_load(a32_at(H, H_ERROR_CODE));
    if (ec != 0) {
      err_post(H, ec);
    }
  }
  a32_store(a32_at(H, H_CURSOR), 0);
}

// Corpus
// ======

// The cores map 8 GiB at a high base and double it in place; the banks move
// up past the pages. The GPU maps its whole span once.

static u64 corpus_size;

static bool corpus_at(void* at, u64 size) {
  void* p = pool_try(at, size);
  if (p != MAP_FAILED && p != at) {
    munmap(p, size);
  }
  return p == at;
}

static void* corpus_map(u64 size) {
  for (u64 hint = 1ull << 45; hint > size; hint /= 2) {
    if (corpus_at((void*)hint, size)) {
      return (void*)hint;
    }
  }
  return pool_mmap(size);
}

static void corpus_lay(u64* H, u64 size) {
  u64 span = size / 8;
  u64 cap  = span > HEAP_OFF ? (span - HEAP_OFF) / (PAGE_LEN + 10) : 0;
  if (cap <= CUBE) {
    err_fail("the GPU span is under the rings, stacks and a page per lane");
  }
  cap = cap < ~0u ? cap : ~0u - 1;
  u64 at = HEAP_OFF + (cap << PAGE_BITS);
  for (u32 c = 0; c < NCLS_ALL; c += 1) {
    Bank* b = bank_at(H, c);
    memcpy(H + at, H + b->off, b->wr * sizeof(u64));
    b->off  = at;
    at     += 2 * (cap >> ((c < NCLS ? NCLS : c) - PAGE_BITS));
  }
  corpus_size = size;
  a32_store_rel(a32_at(H, H_CAP), (u32)cap);
}

static bool corpus_grow(u64* H, u64 need) {
  bool ok = true;
  LOCK(bank_lock);
  while (ok && need > a32_load(a32_at(H, H_CAP))) {
    u64 more = corpus_size;
    ok = !io_gpu && more < 1ull << 43 && corpus_at((char*)H + more, more);
    if (ok) {
      corpus_lay(H, more * 2);
    }
  }
  UNLOCK(bank_lock);
  return ok;
}

static u64* corpus_setup(bool gpu, long threads, u64 bytes) {
  io_gpu     = gpu;
  KEEP_WORDS = gpu ? CHUNK : CAP_WORDS;
  u64 size   = (!gpu ? 1ull << 33 : bytes != 0 ? bytes : gpu_span())
    & ~16383ull;
  CORPUS     = gpu ? gpu_map(size) : corpus_map(size);
  u64* H     = CORPUS;
  corpus_lay(H, size);
  memcpy(H + STAT_OFF, STAT_IMG, STAT_LEN * sizeof(u64));
  a32_store(a32_at(H, H_BUMP), 1);
  if (gpu) {
    gpu_load(size);
  }
  pool_size = threads < 1 ? 1 : threads < CUBE_T ? threads : CUBE_T;
  return H;
}

OUTLINE Term corpus_eval(u64* H, Term t) {
  Env  e = { H, ALC[0] };
  Term rv[WL_RESW];
  while ((t = work_loop(e, io_stk, t, !BANGS && pool_size == 1)) != 0) {
    u64 tl = task_tail(t);
    if ((u32)H[tl + 1] != 0) {
      task_deal(H, t, 0, 0, NULL);
      pool_open();
      cube_run(H, false);
      break;
    }
    if (io_gpu && FID_T[term_aux(t)].bangs) {
      Term cont = H[tl];
      u32  idx  = (u32)(H[tl + 1] >> 32) & 0xFFFF;
      H[tl]     = TERM_HOLE;
      a32_store(a32_at(H, H_CURSOR), 1);
      ring_push(H, 0, t);
      cube_run(H, true);
      t = task_deliver(H, cont, idx, rv, root_take(H, rv));
      if (t == 0) {
        break;
      }
    }
  }
  if (!root_done(H)) {
    err_fail("a delivery lost");
  }
  root_take(H, rv);
  return rv[0];
}

// Io
// ==

// Base's opaque, linear handles pack host fds or pointers into aux and loc.
// Text is UTF-8, decoded as WHATWG does. io_sys_addr refuses a leading zero
// (octal to some libcs). macOS poll misses FIFO EOF, so io_wait selects,
// sized to the highest fd.

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define IO_PARK TERM_HOLE

#define io_hand(v)   term_make(TAG_PAK, (u64)(v) >> 40, (u64)(v) & LOC_MASK)
#define io_hand_v(t) (((u64)term_aux(t) << 40) | term_loc(t))

typedef struct IoWork IoWork;
typedef void (*IoCall)(IoWork* w);
typedef Term (*IoPack)(Env e, IoWork* w);

struct IoWork {
  intptr_t hand;
  intptr_t made;
  u32      word;
  u64      size;
  char*    data;
  char*    text;
  u32      code;
  IoCall   call;
  IoPack   pack;
  Term     cont;
  Term     item;
  u64      time;
  short    evts;
  IoWork*  next;
  IoWork*  prev;
};

typedef Term (*Effect)(Env e, Term* f, IoWork* w);

Effect io_eff_rows[sizeof CID_T / sizeof *CID_T];
static u32    io_live;

OUTLINE void* io_mem(void* mem) {
  if (mem == NULL) {
    err_fail("host allocation failed");
  }
  return mem;
}

static int io_sys_addr(const char* host, u32 port, struct sockaddr_in* at) {
  *at = (struct sockaddr_in){ .sin_family = AF_INET,
    .sin_port = htons((uint16_t)port) };
  for (const char* p = host; *p != 0; p += 1) {
    if ((p == host || p[-1] == '.') && *p == '0'
      && p[1] >= '0' && p[1] <= '9') {
      return -1;
    }
  }
  return port > 65535 || inet_pton(AF_INET, host, &at->sin_addr) != 1
    ? -1 : 0;
}

static int    io_argc;
static char** io_argv;

static void io_eff(u32 cid, Effect run) {
  if (io_eff_rows[cid] != NULL) {
    err_fail("two effects register one request");
  }
  io_eff_rows[cid] = run;
}

static u64 io_sys_end(IoWork* w, ssize_t n) {
  w->code = n < 0 ? (u32)errno : 0;
  return n < 0 ? 0 : (u64)n;
}

static IoWork* io_runs;
static IoWork* io_park;
static IoWork* io_jobs;

static void io_push(IoWork** q, IoWork* a) {
  IoWork* l = *q != NULL ? *q : a;
  a->next = l->next;
  l->next = a;
  *q      = a;
}

static IoWork* io_pop(IoWork** q) {
  IoWork* a  = (*q)->next;
  (*q)->next = a->next;
  *q         = a != *q ? *q : NULL;
  return a;
}

static void io_spawn(Term m) {
  IoWork* a = io_mem(calloc(1, sizeof(IoWork)));
  a->cont  = m;
  a->item  = term_clo(FID_IO_emit, 0);
  io_push(&io_runs, a);
  io_live += 1;
}

// io_park stays in deadline order (time 0, none, sorts last; ties keep
// their park order).
static void io_park_add(IoWork* w) {
  IoWork* p = io_park;
  if (p == NULL || p->time - 1 <= w->time - 1) {
    io_push(&io_park, w);
  } else {
    while (p->next->time - 1 <= w->time - 1) {
      p = p->next;
    }
    io_push(&p, w);
  }
  w->prev       = w->next->prev;
  w->next->prev = w;
}

static void io_park_cut(IoWork* w) {
  w->prev->next = w->next;
  w->next->prev = w->prev;
  io_park = w->next == w ? NULL : io_park == w ? w->prev : io_park;
}

#define io_parked(a) for (IoWork* a = io_park; a != NULL; \
  a = a->next != io_park ? a->next : NULL)

static Term io_wait_on(IoWork* w, int fd, short evts, u64 time, IoPack more) {
  w->word = (u32)fd;
  w->pack = more;
  w->time = time;
  w->evts = evts;
  io_park_add(w);
  return IO_PARK;
}

static void io_lost(void) {
  if (errno == EPIPE) {
    signal(SIGPIPE, SIG_DFL);
    raise(SIGPIPE);
  }
  err_fail("a short write on a standard stream");
}

OUTLINE void io_out(FILE* h, const char* data, u64 len) {
  if (fwrite(data, 1, len, h) != len) {
    io_lost();
  }
}

OUTLINE void io_sync(void) {
  if (fflush(stdout) != 0) {
    io_lost();
  }
}

static u64 io_utf8(char* buf, u64 c) {
  c = c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF) ? 0xFFFD : c;
  u64 k = c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  for (u64 i = k; i > 1; i -= 1) {
    buf[i - 1] = (char)(0x80 | (c & 0x3F));
    c >>= 6;
  }
  buf[0] = (char)(k == 1 ? c : (0xF00 >> k) | c);
  return k;
}

// io_cbuf writes a String (cons SCon) as UTF-8, or a List (cons Con) as
// its bytes, with no UTF-8: NULL if a value is past 255.
OUTLINE char* io_cbuf(Env e, Term s, u64* len, u64 cons) {
  u64   cap = 64, n = 0, bad = 0;
  char* buf = io_mem(malloc(cap));
  while (term_aux(s) == cons) {
    Term fb[2];
    spare_free(e, cls_fit(2), ctr_take(e, s, 2, fb));
    if (n + 5 > cap) {
      cap *= 2;
      buf = io_mem(realloc(buf, cap));
    }
    if (cons == CID_SCon) {
      n += io_utf8(buf + n, fb[0]);
    } else {
      bad |= fb[0] > 255;
      buf[n++] = (char)fb[0];
    }
    s = fb[1];
  }
  buf[n] = 0;
  *len = n;
  if (bad) {
    free(buf);
    return NULL;
  }
  return buf;
}

#define io_cstr(e, s, len) io_cbuf(e, s, len, CID_SCon)

OUTLINE void io_errs(Env e, Term s) {
  u64   n;
  char* text = io_cstr(e, s, &n);
  io_sync();
  text[n] = '\n';
  io_out(stderr, text, n + 1);
  free(text);
}

#define io_nul(s, n) (strlen(s) != (n))

#define io_seal(e, t, cid) (CID_T[cid].hot ? rfc_seal(e, t) : (t))

static Term io_node(Env e, u64 cid, Term a, Term b) {
  u64 l = heap_alloc(e, 1);
  e.mem[l]     = io_seal(e, a, cid);
  e.mem[l + 1] = io_seal(e, b, cid);
  return term_ctr(cid, l);
}

static Term io_str(Env e, const char* p, u64 n) {
  Term  s;
  Term* hole = &s;
  u64  c = 0, need = 0, lo = 0x80, hi = 0xBF;
  for (u64 i = 0; i < n || need > 0; i += 1) {
    u64 b = i < n ? (uint8_t)p[i] : 0x100;
    if (need > 0 && (b < lo || b > hi)) {
      need = 0;
      c    = 0xFFFD;
      i   -= 1;
    } else if (need > 0) {
      lo = 0x80;
      hi = 0xBF;
      c  = (c << 6) | (b & 0x3F);
      if (--need > 0) {
        continue;
      }
    } else if (b < 0x80) {
      c = b;
    } else if (b < 0xC2 || b > 0xF4) {
      c = 0xFFFD;
    } else {
      need = b < 0xE0 ? 1 : b < 0xF0 ? 2 : 3;
      lo   = b == 0xE0 ? 0xA0 : b == 0xF0 ? 0x90 : 0x80;
      hi   = b == 0xED ? 0x9F : b == 0xF4 ? 0x8F : 0xBF;
      c    = b & (0x3F >> need);
      continue;
    }
    u64  l = heap_alloc(e, 1);
    Term t = term_ctr(CID_SCon, l);
    e.mem[l] = c;
    *hole = hole == &s ? t : io_seal(e, t, CID_SCon);
    hole = &e.mem[l + 1];
  }
  *hole = term_pak(CID_SNil, 0);
  return s;
}

#ifdef CID_Con

static Term io_list(Env e, const char* p, u64 n) {
  Term xs = term_pak(CID_Nil, 0);
  for (u64 i = n; i > 0; i -= 1) {
    xs = io_node(e, CID_Con, (uint8_t)p[i - 1], xs);
  }
  return xs;
}

#endif

#define io_tup(e, a, b) io_node(e, CID_Tuple, a, b)
#define io_done(e, v)   io_box(e, CID_Done, v)
#define io_res(e, w, v) ((w)->code ? io_fail(e, (w)->code, NULL) \
  : io_done(e, v))
#define io_until(ms)    (io_tick() + (u64)(ms) * 1000000ull)
#define io_again(w) ((w)->code == EAGAIN \
  && ((w)->time == 0 || io_tick() < (w)->time))
#define io_poll_end(e, w, rest, r) ((w)->code == EAGAIN \
  ? io_box(e, CID_Wait, rest) : (w)->time ? io_box(e, CID_Ready, r) : (r))

static Term io_box(Env e, u64 cid, Term v) {
  u64 l = heap_alloc(e, 0);
  e.mem[l] = io_seal(e, v, cid);
  return term_ctr(cid, l);
}

static Term io_err(Env e, u32 code, const char* text) {
  const char* s = text != NULL ? text : strerror((int)code);
  return io_tup(e, code, io_str(e, s, strlen(s)));
}

#define io_fail(e, code, text) io_box(e, CID_Fail, io_err(e, code, text))

static pthread_mutex_t io_gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  io_bell = PTHREAD_COND_INITIALIZER;
static u32             io_busy;
static u32             io_size;
static int             io_wake_fd[2];

static void io_take(Env e) {
  IoWork* acts[64];
  ssize_t n;
  while ((n = read(io_wake_fd[0], acts, sizeof acts)) > 0) {
    for (u32 i = 0; i < (u32)n / sizeof(IoWork*); i += 1) {
      IoWork* a = acts[i];
      a->item   = a->pack(e, a);
      io_push(&io_runs, a);
      io_busy -= 1;
    }
  }
}

static void* io_help(void* arg) {
  for (;;) {
    pthread_mutex_lock(&io_gate);
    while (io_jobs == NULL) {
      pthread_cond_wait(&io_bell, &io_gate);
    }
    IoWork* a = io_pop(&io_jobs);
    pthread_mutex_unlock(&io_gate);
    a->call(a);
    while (write(io_wake_fd[1], &a, sizeof a) != sizeof a) {
    }
  }
}

static Term io_work(IoWork* w, IoCall call, IoPack pack) {
  w->call  = call;
  w->pack  = pack;
  io_busy += 1;
  if (io_busy > io_size && io_size < IO_HELP) {
    pthread_t tid;
    if (pthread_create(&tid, NULL, io_help, NULL)) {
      err_fail("pthread_create");
    }
    pthread_detach(tid);
    io_size += 1;
  }
  pthread_mutex_lock(&io_gate);
  io_push(&io_jobs, w);
  pthread_cond_signal(&io_bell);
  pthread_mutex_unlock(&io_gate);
  return IO_PARK;
}

static bool io_bit(u8* set, int fd, bool put) {
  u8* at = set + fd / 8;
  *at |= put << fd % 8;
  return *at >> fd % 8 & 1;
}

static void io_wait(Env e, bool block) {
  int top  = io_wake_fd[0];
  u64 soon = io_park != NULL ? io_park->next->time : 0;
  io_parked(a) {
    if (a->evts != 0 && (int)a->word > top) {
      top = (int)a->word;
    }
  }
  u64 len = (u64)top / 64 * 8 + 8;
  u8* set = io_mem(calloc(2, len));
  io_bit(set, io_wake_fd[0], true);
  io_parked(a) {
    if (a->evts != 0) {
      io_bit(set + (a->evts == POLLOUT) * len, (int)a->word, true);
    }
  }
  u64 tick = io_tick();
  u64 ms = soon > tick && block ? (soon - tick) / 1000000 + 1 : 0;
  struct timeval tv = { ms / 1000, ms % 1000 * 1000 };
  io_sync();
  if (select(top + 1, (fd_set*)set, (fd_set*)(set + len), NULL,
    soon == 0 && block ? NULL : &tv) < 0) {
    if (errno != EINTR) {
      err_fail("the poller failed");
    }
    memset(set, 0, 2 * len);
  }
  if (io_bit(set, io_wake_fd[0], false)) {
    io_take(e);
  }
  u64     now  = io_tick();
  IoWork* todo = io_park;
  io_park = NULL;
  while (todo != NULL) {
    IoWork* a   = io_pop(&todo);
    bool    due = (a->evts != 0
        && io_bit(set + (a->evts == POLLOUT) * len, (int)a->word, false))
      || (a->time != 0 && a->time <= now);
    if (!due) {
      io_park_add(a);
      continue;
    }
    Term x = a->pack(e, a);
    if (x != IO_PARK) {
      a->item = x;
      io_push(&io_runs, a);
    }
  }
  free(set);
}

static int f32_text(char* buf, f32 v) {
  int n = 0;
  int p = 0;
  if (v != v) {
    return sprintf(buf, "nan");
  }
  for (; p < 9; p += 1) {
    n = snprintf(buf, 40, "%.*e", p, (double)v);
    if (strtof(buf, NULL) == v) {
      break;
    }
  }
  char* ep = strchr(buf, 'e');
  if (ep == NULL) {
    return n;
  }
  int ex = atoi(ep + 1);
  if (ex >= 21 || ex <= -7) {
    return (int)(ep - buf) + sprintf(ep, "e%+d", ex);
  }
  // The scientific form has the rounded digits d, d[i] at the place ex - i:
  // write each place from the highest down, with a point before place -1.
  int  s = *buf == '-';
  char d[9];
  d[0] = buf[s];
  memcpy(d + 1, buf + s + 2, p);
  n = s;
  for (int q = ex > 0 ? ex : 0; q >= 0 || q >= ex - p; q -= 1) {
    if (q == -1) {
      buf[n++] = '.';
    }
    buf[n++] = q <= ex && q >= ex - p ? d[ex - q] : '0';
  }
  buf[n] = 0;
  return n;
}

static Term f32_show(Env e, Term x) {
  char buf[40];
  return io_str(e, buf, f32_text(buf, f32_unbox(x)));
}

static Term f32_read(Env e, Term s) {
  u64 n;
  char* text = io_cstr(e, s, &n);
  char* end;
  f32 v = strtof(text, &end);
  Term out = n > 0 && (u64)(end - text) == n && strpbrk(text, "xX(") == NULL
    ? io_box(e, CID_Some, f32_rewrap(v)) : term_pak(CID_None, 0);
  free(text);
  return out;
}


// Run
// ===

static void io_step(Env e, IoWork* a) {
  for (u32 step = 0; step < 64; step += 1) {
    u64  ap  = task_node(e, FID_Clo_apply, TERM_HOLE, 0, 0);
    e.mem[ap]     = a->cont;
    e.mem[ap + 1] = a->item;
    Term req = corpus_eval(e.mem, term_tsk(FID_Clo_apply, ap));
    u32  c   = term_aux(req);
    if (c == CID_Emit) {
      term_drop(e, req);
      free(a);
      io_live -= 1;
      return;
    }
    Term fs[256];
    u32  n = cid_arity(c);
    if (term_tag(req) == TAG_PAK) fs[0] = term_loc(req);
    else spare_free(e, cls_fit(n), ctr_take(e, req, n, fs));
    if (c == CID_Halt || c == CID_Exit) {
      if (c == CID_Halt) io_errs(e, fs[1]);
      if ((u32)fs[0] > 255) err_fail("exit code must be in 0..255");
      io_sync();
      exit((int)(u32)fs[0]);
    }
    if (io_eff_rows[c] == NULL) {
      err_fail("an alien request");
    }
    a->cont = fs[n - 1];
    Term x  = io_eff_rows[c](e, fs, a);
    if (x == IO_PARK) {
      return;
    }
    a->item = x;
  }
  io_push(&io_runs, a);
}

OUTLINE void io_loop(u64* H) {
  Env e = { H, ALC[0] };
  io_stk = pool_stack();
  signal(SIGPIPE, SIG_IGN);
  if (pipe(io_wake_fd) || fcntl(io_wake_fd[0], F_SETFL, O_NONBLOCK)) {
    err_fail("the event loop failed to open");
  }
  Term m = corpus_eval(H, term_tsk(MAIN_FID, task_node(e, MAIN_FID,
    TERM_HOLE, 0, 0)));
  io_spawn(m);
  u64 look = 0;
  for (u32 n = 0;; n += 1) {
    if (io_runs == NULL) {
      if (io_live == 0) {
        return;
      }
      if (io_park == NULL && io_busy == 0) {
        io_sync();
        err_fail("deadlock: every computation waits on a channel");
      }
      io_wait(e, true);
      continue;
    }
    if ((n & 63) == 0) {
      if (io_busy != 0) {
        io_take(e);
      }
      if (io_park != NULL) {
        u64 now = io_tick();
        if (now >= look || io_park->next->time - 1 < now) {
          look = now + 10000000;
          io_wait(e, false);
        }
      }
    }
    io_step(e, io_pop(&io_runs));
  }
}

// Requests
// ========

// Host runtime admission and actual Metal command-buffer timeline metadata.
static Term metal_backend_run(Env e, Term* f, IoWork* w) {
#if BEND_METAL
  return (Term)(io_gpu && gpu_dev != nil);
#else
  return 0;
#endif
}
static Term metal_events_run(Env e, Term* f, IoWork* w) {
#if BEND_METAL
  return (Term)gpu_ev_at;
#else
  return 0;
#endif
}
static Term metal_heap_mib_run(Env e, Term* f, IoWork* w) {
  return (Term)(corpus_size >> 20);
}
static void __attribute__((constructor)) metal_runtime_use(void) {
#ifdef CID_metal_runtime_backend
  io_eff(CID_metal_runtime_backend, metal_backend_run);
#endif
#ifdef CID_metal_runtime_events
  io_eff(CID_metal_runtime_events, metal_events_run);
#endif
#ifdef CID_metal_runtime_heap_mib
  io_eff(CID_metal_runtime_heap_mib, metal_heap_mib_run);
#endif
}
// IO
// ==

// str makes an argument's value from its bytes: io_str's String, or
// io_list's bytes for IO.args_bytes.
static Term io_args_with(Env e, Term (*str)(Env e, const char* p, u64 n)) {
  Term xs = term_pak(CID_Nil, 0);
  for (int i = io_argc; i > 0; i -= 1) {
    const char* a = io_argv[i - 1];
    xs = io_node(e, CID_Con, str(e, a, strlen(a)), xs);
  }
  return xs;
}

#ifdef CID_IO_args

Term io_args_run(Env e, Term* f, IoWork* w) {
  return io_args_with(e, io_str);
}

static void __attribute__((constructor)) io_args_use(void) {
  io_eff(CID_IO_args, io_args_run);
}

#endif

#ifdef CID_IO_args_bytes

Term io_args_bytes_run(Env e, Term* f, IoWork* w) {
  return io_args_with(e, io_list);
}

static void __attribute__((constructor)) io_args_bytes_use(void) {
  io_eff(CID_IO_args_bytes, io_args_bytes_run);
}

#endif
// IO
// ==

Term io_print_run(Env e, Term* f, IoWork* w) {
  uint64_t n = 0;
  char* text = io_cstr(e, f[0], &n);
  text[n] = '\n';
  io_out(stdout, text, n + 1);
  free(text);
  return term_pak(CID_Unit, 0);
}

static void __attribute__((constructor)) io_print_use(void) {
  io_eff(CID_IO_print, io_print_run);
}


// Main
// ====

int main(int argc, char** argv) {
  long   thr = cpu_count();
  int    gpu = -1;
  u64    mem = 0;
  char*  env = strdup(getenv("BENDRTS") != NULL ? getenv("BENDRTS") : "");
  char** rts = malloc((argc + strlen(env) + 1) * sizeof(char*));
  int    nr  = 0;
  bool   in  = false;
  for (char* w = strtok(env, " \t\n"); w != NULL; w = strtok(NULL, " \t\n")) {
    rts[nr++] = w;
  }
  rts[nr++] = NULL;
  io_argv = argv;
  io_argc = argc > 0;
  for (int i = 1; i < argc; i += 1) {
    char* a = argv[i];
    if (!strcmp(a, "--RTS")) {
      for (i += 1; i < argc; i += 1) {
        io_argv[io_argc++] = argv[i];
      }
    } else if (!strcmp(a, "+RTS") || !strcmp(a, "-RTS")) {
      in        = a[0] == '+';
      rts[nr++] = NULL;
    } else if (in) {
      rts[nr++] = a;
    } else {
      io_argv[io_argc++] = a;
    }
  }
  for (int i = 0; i < nr; i += 1) {
    const char* a = rts[i];
    const char* v = i + 1 < nr && rts[i + 1] != NULL ? rts[i + 1] : "";
    char*       end;
    if (a == NULL) {
      continue;
    } else if (!strcmp(a, "--help")) {
      printf(CLI_HELP, argv[0]);
      return 0;
    } else if (!strcmp(a, "--gpu-build")) {
      if (gpu_probe() == NULL && !gpu_make(gpu_path())) {
        fprintf(stderr, "bend: cannot write %s\n", gpu_path());
        return 1;
      }
      return 0;
    } else if (!strcmp(a, "--threads")) {
      thr = strtol(v, NULL, 10);
      if (thr < 1 || v[strspn(v, "0123456789")] != 0) {
        err_fail("expected a thread count of 1 or more after --threads");
      }
      i += 1;
    } else if (!strcmp(a, "--gpu")) {
      double n   = strtod(v, &end);
      u64    mul = !strcmp(end, "GB") ? 1ull << 30
        : !strcmp(end, "MB") ? 1ull << 20 : 0;
      mem = (u64)(n * (double)mul);
      if (!strcmp(v, "off")) {
        gpu = 0;
      } else if (!strcmp(v, "on")
        || (mem >= 16384 && end == v + strspn(v, "0123456789."))) {
        gpu = 1;
      } else {
        err_fail("expected on, off or a size like 4GB after --gpu");
      }
      i += 1;
    } else {
      fprintf(stderr, "bend: unknown runtime option %s\n", a);
      return 1;
    }
  }
  const char* why = gpu != 0 && BANGS != 0 ? gpu_probe() : "";
  if (gpu == 1 && BANGS != 0 && why != NULL) {
    err_fail(why);
  }
  io_loop(corpus_setup(why == NULL, thr, mem));
  io_sync();
  return 0;
}

#endif
