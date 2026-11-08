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
#ifdef CID(backend)
  io_eff(CID(backend), metal_backend_run);
#endif
#ifdef CID(events)
  io_eff(CID(events), metal_events_run);
#endif
#ifdef CID(heap_mib)
  io_eff(CID(heap_mib), metal_heap_mib_run);
#endif
}
