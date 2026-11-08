import Base
import ./ir.bend as Ir
import ./metal_runtime.bend as Metal

def nodes() -> +List<Ir.IExpr>:
  [Ir.ISlot{0}, Ir.IInt{1}, Ir.IAdd{0,1}]

def calculate(id: U32) -> Ir.Val:
  Ir.eval(nodes(), Ir.EvalEnv{[Ir.VI{2}], Nil{}}, id)

def finish(result: Ir.Val, backend: U32) -> IO(Unit):
  match result:
    case Ir.VI{number}:
      do IO<Unit>:
        events : U32 <- Metal.events()
        heap : U32 <- Metal.heap_mib()
        IO.print("{\"backend\":" ++ U32.show(backend) ++ ",\"events\":" ++ U32.show(events) ++ ",\"heap\":" ++ U32.show(heap) ++ ",\"value\":" ++ U32.show(number) ++ "}")
    case _: IO.die(Unit, 2, "unexpected IR value")

def parsed(result: Maybe<&2,U32>, backend: U32) -> IO(Unit):
  match result:
    case Some{id}: finish(calculate!(id), backend)
    case None{}: IO.die(Unit, 2, "expected opcode index")

def args(input: List<String>, backend: U32) -> IO(Unit):
  match input:
    case _ <> id <> Nil{}: parsed(U32.read(id), backend)
    case _: IO.die(Unit, 2, "expected opcode index")

def main() -> IO(Unit):
  do IO<Unit>:
    backend : U32 <- Metal.backend()
    input : List<String> <- IO.args()
    args(input, backend)
