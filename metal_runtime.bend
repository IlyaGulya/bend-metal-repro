import Base

# Platform introspection only. All TLA+ parsing, IR evaluation and search stay Bend.
def backend() -> IO(U32):
  import "./metal_runtime.c"
  import "./metal_runtime.js"

def events() -> IO(U32):
  import "./metal_runtime.c"
  import "./metal_runtime.js"

def heap_mib() -> IO(U32):
  import "./metal_runtime.c"
  import "./metal_runtime.js"
