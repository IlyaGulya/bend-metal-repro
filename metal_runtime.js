// The JS target has no Metal device; admission must fail, never emulate it.
if (typeof CID(backend) !== "undefined") io_eff(CID(backend), () => 0);
if (typeof CID(events) !== "undefined") io_eff(CID(events), () => 0);
if (typeof CID(heap_mib) !== "undefined") io_eff(CID(heap_mib), () => 0);
