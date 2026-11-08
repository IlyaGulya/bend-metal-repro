import Base
import ./checker.bend as Core

# Resolved, lowered finite-model IR.
#
# The source AST is never consulted at run time. Expressions produce structured
# finite values over flat state cells and numeric lexical slots. Packed fixed
# functions retain one cell per key; first-class functions, records and tuples
# share balanced trees. Actions carry a current-state guard and either complete
# cell assignments or explicit finite numeric-local choices and constraints.
# Signed-32 integers retain their two's-complement bits.

type Val is Data:
  VI{v: U32}
  VB{b: Bool}
  VS{s: String}
  VSet{items: +List<Val>, index: Domain}
  VFun{tree: FunctionTree, sequence: Bool}
  VE{msg: String}

# Runtime domains are indexed once during lowering, never scanned per state.
type DomainTree is Data:
  DomainEmpty{}
  DomainNode{left: DomainTree, value: Val, right: DomainTree, size: U32}

type Domain is Data:
  IntDomain{first: U32, size: U32}
  TreeDomain{tree: DomainTree}

type DomainBuilt is Data:
  DomainBuilt{tree: DomainTree, rest: +List<Val>}

# Records, tuples and finite functions share one extensional representation.
# Existing-key replacement and sequence append copy only a logarithmic path.
type FunctionTree is Data:
  FunctionEmpty{}
  FunctionNode{left: FunctionTree, key: Val, value: Val, right: FunctionTree, size: U32, height: U32}

type FunctionBuilt is Data:
  FunctionBuilt{tree: FunctionTree, keys: +List<Val>, values: +List<Val>}

type Cell is Data:
  Cell{owner: U32, key: +Maybe<Val>, dom: Domain, place: U32}

type Var is Data:
  Var{name: String, start: U32, count: U32, function: Bool}

type Assign is Data:
  Assign{slot: U32, expr: U32}

type Case is Data:
  Case{assigns: +List<Assign>}

type Action is Data:
  Action{guard: U32, assigns: +List<Assign>}
  Relation{guard: U32, choices: +List<Choice>, values: +List<U32>, predicate: U32}

type Choice is Data:
  Choice{slot: U32, expr: U32, membership: Bool}

type ChoiceResult is Data:
  ChoiceResult{reversed: +List<Core.Edge>, fuel: U32, failed: Bool}

type Projection is Data:
  Projection{cells: +List<Cell>, reversed: +List<Val>, valid: Bool}

type Named is Data:
  Named{name: String, expr: U32}

type Local is Data:
  Local{slot: U32, value: Val}

type EvalEnv is Data:
  EvalEnv{state: +List<Val>, locals: +List<Local>}

type IExpr is Data:
  IInt{v: U32}
  IBool{b: Bool}
  IStr{s: String}
  ISlot{slot: U32}
  IValue{value: Val}
  ILocal{slot: U32}
  ISet{items: +List<U32>}
  ISetBinary{op: U32, left: U32, right: U32}
  ISetUnary{op: U32, value: U32}
  IBounded{op: U32, slot: U32, domain: U32, body: U32}
  IFunction{keys: +List<Val>, values: +List<U32>}
  IProduct{keys: +List<Val>, domains: +List<U32>}
  ILookup{function: U32, key: U32}
  IUpdate{function: U32, key: U32, slot: U32, value: U32}
  IDomain{function: U32}
  ISequence{op: U32, args: +List<U32>}
  IAdd{l: U32, r: U32}
  ISub{l: U32, r: U32}
  INeg{a: U32}
  IEq{l: U32, r: U32}
  INe{l: U32, r: U32}
  ILt{l: U32, r: U32}
  ILe{l: U32, r: U32}
  IGt{l: U32, r: U32}
  IGe{l: U32, r: U32}
  IAnd{l: U32, r: U32}
  IOr{l: U32, r: U32}
  INot{a: U32}
  IImp{l: U32, r: U32}
  IIte{c: U32, t: U32, e: U32}
  IBad{msg: String}

# --------------------------------------------------------- scalar primitives

def negative(a: U32) -> Bool:
  U32.is_ge(a, 2147483648)

def pickVal(c: Bool, yes: Val, no: Val) -> Val:
  match c:
    case True{}:
      yes
    case False{}:
      no

def pickText(c: Bool, yes: String, no: String) -> String:
  match c:
    case True{}:
      yes
    case False{}:
      no

def sadd(+a: U32, +b: U32) -> Val:
  +s = U32.add(a, b)
  same = Bool.not(Bool.xor(negative(a), negative(b)))
  flipped = Bool.xor(negative(s), negative(a))
  pickVal(Bool.and(same, flipped), VE{"signed 32-bit addition overflow"}, VI{s})

def sneg(+a: U32) -> Val:
  pickVal(U32.is_eq(a, 2147483648), VE{"signed 32-bit negation overflow"}, VI{U32.sub(0, a)})

def ssub(+a: U32, +b: U32) -> Val:
  +difference = U32.sub(a, b)
  different_signs = Bool.xor(negative(a), negative(b))
  changed_sign = Bool.xor(negative(difference), negative(a))
  pickVal(Bool.and(different_signs, changed_sign), VE{"signed 32-bit subtraction overflow"}, VI{difference})

@unsafe
def slt(+a: U32, +b: U32) -> Bool:
  slt_go(negative(a), negative(b), a, b)

def slt_go(na: Bool, nb: Bool, a: U32, b: U32) -> Bool:
  match na nb:
    case True{} False{}:
      True{}
    case False{} True{}:
      False{}
    case _ _:
      U32.is_lt(a, b)

def sle(+a: U32, +b: U32) -> Bool:
  Bool.or(slt(a, b), U32.is_eq(a, b))

def sgt(a: U32, b: U32) -> Bool:
  slt(b, a)

def sge(a: U32, b: U32) -> Bool:
  sle(b, a)

def boolEq(a: Bool, b: Bool) -> Bool:
  match a b:
    case True{} True{}:
      True{}
    case False{} False{}:
      True{}
    case _ _:
      False{}

def textEq(a: String, b: String) -> Bool:
  String.eq(a, b)

@unsafe
def listEq(xs: +List<Val>, ys: +List<Val>) -> Bool:
  match xs ys:
    case Nil{} Nil{}:
      True{}
    case Nil{} _:
      False{}
    case _ Nil{}:
      False{}
    case h <> t v <> u:
      Bool.and(veq(h, v), listEq(t, u))

@unsafe
def veq(a: Val, b: Val) -> Bool:
  match a b:
    case VI{x} VI{y}:
      U32.is_eq(x, y)
    case VB{x} VB{y}:
      boolEq(x, y)
    case VS{x} VS{y}:
      String.eq(x, y)
    case VSet{xs, ix} VSet{ys, iy}:
      Bool.and(U32.is_eq(domainSize(ix), domainSize(iy)), listEq(xs, ys))
    case VFun{left, _} VFun{right, _}:
      functionEqual(functionOrder(left, right))
    case _ _:
      False{}

def vadd(a: Val, b: Val) -> Val:
  match a b:
    case VE{msg} _:
      VE{msg}
    case _ VE{msg}:
      VE{msg}
    case VI{x} VI{y}:
      sadd(x, y)
    case _ _:
      VE{"expected integers for +"}

def vsub(a: Val, b: Val) -> Val:
  match a b:
    case VE{msg} _:
      VE{msg}
    case _ VE{msg}:
      VE{msg}
    case VI{x} VI{y}:
      ssub(x, y)
    case _ _:
      VE{"expected integers for -"}

def vneg(a: Val) -> Val:
  match a:
    case VE{msg}:
      VE{msg}
    case VI{x}:
      sneg(x)
    case _:
      VE{"expected an integer for unary -"}

def vlt(a: Val, b: Val) -> Val:
  match a b:
    case VE{msg} _:
      VE{msg}
    case _ VE{msg}:
      VE{msg}
    case VI{x} VI{y}:
      VB{slt(x, y)}
    case _ _:
      VE{"expected integers for <"}

def vle(a: Val, b: Val) -> Val:
  match a b:
    case VE{msg} _:
      VE{msg}
    case _ VE{msg}:
      VE{msg}
    case VI{x} VI{y}:
      VB{sle(x, y)}
    case _ _:
      VE{"expected integers for <="}

def vgt(a: Val, b: Val) -> Val:
  vlt(b, a)

def vge(a: Val, b: Val) -> Val:
  vle(b, a)

def veqb(a: Val, b: Val) -> Val:
  match a b:
    case VE{message} _:
      VE{message}
    case _ VE{message}:
      VE{message}
    case x y:
      VB{veq(x, y)}


def vand(a: Val, b: Val) -> Val:
  match a b:
    case VE{msg} _:
      VE{msg}
    case _ VE{msg}:
      VE{msg}
    case VB{x} VB{y}:
      VB{Bool.and(x, y)}
    case _ _:
      VE{"expected booleans for /\\"}

def vor(a: Val, b: Val) -> Val:
  match a b:
    case VE{msg} _:
      VE{msg}
    case _ VE{msg}:
      VE{msg}
    case VB{x} VB{y}:
      VB{Bool.or(x, y)}
    case _ _:
      VE{"expected booleans for \\/"}

def vnot(a: Val) -> Val:
  match a:
    case VE{msg}:
      VE{msg}
    case VB{x}:
      VB{Bool.not(x)}
    case _:
      VE{"expected a boolean for ~"}
def vneb(a: Val, b: Val) -> Val:
  vnot(veqb(a, b))

def vimp(a: Val, b: Val) -> Val:
  vor(vnot(a), b)

def vite(c: Val, yes: Val, no: Val) -> Val:
  match c:
    case VE{msg}:
      VE{msg}
    case VB{x}:
      pickVal(x, yes, no)
    case _:
      VE{"expected a boolean condition"}

# ------------------------------------------------------------------ rendering

def boolText(b: Bool) -> String:
  match b:
    case True{}:
      "true"
    case False{}:
      "false"

def signedText(+v: U32) -> String:
  pickText(negative(v), "-" ++ U32.show(U32.sub(0, v)), U32.show(v))

@unsafe
def showVal(v: Val) -> String:
  match v:
    case VI{x}:
      signedText(x)
    case VB{b}:
      boolText(b)
    case VS{s}:
      Core.json_quote(s)
    case VSet{items, _}:
      "{\"set\":[" ++ String.join(showSetItems(items), ",") ++ "]}"
    case VFun{tree, _}:
      "{\"function\":[" ++ String.join(functionJson(tree, Nil{}), ",") ++ "]}"
    case VE{msg}:
      Core.json_quote("<evaluation error: " ++ msg ++ ">")

@unsafe
def showSetItems(items: +List<Val>) -> +List<String>:
  match items:
    case Nil{}: Nil{}
    case value <> rest: showVal(value) <> showSetItems(rest)


# ---------------------------------------------------------------- collections

def vlen(xs: +List<Val>) -> U32:
  match xs:
    case Nil{}:
      0
    case _ <> t:
      U32.inc(vlen(t))

def nthVal(xs: +List<Val>, n: Nat, d: Val) -> Val:
  match xs n:
    case Nil{} _:
      d
    case h <> t 0n:
      h
    case _ <> t 1n+p:
      nthVal(t, p, d)

def valueTag(value: Val) -> U32:
  match value:
    case VI{_}: 0
    case VB{_}: 1
    case VS{_}: 2
    case VSet{_, _}: 3
    case VFun{_, _}: 4
    case VE{_}: 5

def signedOrderSigns(a_negative: Bool, b_negative: Bool, a: U32, b: U32) -> Cmp:
  match a_negative b_negative:
    case True{} False{}: LT{}
    case False{} True{}: GT{}
    case _ _: U32.cmp(a, b)

def signedOrder(+a: U32, +b: U32) -> Cmp:
  signedOrderSigns(negative(a), negative(b), a, b)

@unsafe
def valueOrder(+a: Val, +b: Val) -> Cmp:
  match a b:
    case VI{x} VI{y}: signedOrder(x, y)
    case VB{x} VB{y}: U32.cmp(Bool.to_u32(x), Bool.to_u32(y))
    case VS{x} VS{y}: String.order(x, y)
    case VSet{x, ix} VSet{y, iy}: domainPartsOrder(U32.cmp(domainSize(ix), domainSize(iy)), x, y)
    case VFun{x, _} VFun{y, _}: functionOrder(x, y)
    case VE{x} VE{y}: String.order(x, y)
    case _ _: U32.cmp(valueTag(a), valueTag(b))

@unsafe
def valuesOrder(a: +List<Val>, b: +List<Val>) -> Cmp:
  match a b:
    case Nil{} Nil{}: EQ{}
    case Nil{} _: LT{}
    case _ Nil{}: GT{}
    case x <> xs y <> ys: domainPartsOrder(valueOrder(x, y), xs, ys)

@unsafe
def domainPartsOrder(order: Cmp, a: +List<Val>, b: +List<Val>) -> Cmp:
  match order:
    case EQ{}: valuesOrder(a, b)
    case other: other

def valueLe(a: Val, b: Val) -> Bool:
  Cmp.is_le(valueOrder(a, b))

def domainTreeSize(tree: DomainTree) -> U32:
  match tree:
    case DomainEmpty{}: 0
    case DomainNode{_, _, _, size}: size

def domainSize(domain: Domain) -> U32:
  match domain:
    case IntDomain{_, size}: size
    case TreeDomain{tree}: domainTreeSize(tree)

@unsafe
def integerSequence(+previous: U32, values: +List<Val>) -> Bool:
  match values:
    case Nil{}: True{}
    case VI{+value} <> rest:
      integerSequenceStep(Bool.and(U32.is_ne(previous, 2147483647), U32.is_eq(value, U32.inc(previous))), value, rest)
    case _: False{}

@unsafe
def integerSequenceStep(adjacent: Bool, value: U32, rest: +List<Val>) -> Bool:
  match adjacent:
    case True{}: integerSequence(value, rest)
    case False{}: False{}

@unsafe
def makeDomain(+values: +List<Val>) -> Domain:
  domainUnsorted(values, vlen(values))

@unsafe
def domainUnsorted(+values: +List<Val>, +size: U32) -> Domain:
  match values:
    case VI{+first} <> rest:
      domainInterval(integerSequence(first, rest), first, size, values)
    case _: domainSorted(List.sort(~Val, ~valueLe, values), size)

@unsafe
def domainInterval(contiguous: Bool, first: U32, +size: U32, values: +List<Val>) -> Domain:
  match contiguous:
    case True{}: IntDomain{first, size}
    case False{}: domainSorted(List.sort(~Val, ~valueLe, values), size)

@unsafe
def domainSorted(+values: +List<Val>, +size: U32) -> Domain:
  match values:
    case VI{+first} <> rest:
      domainSortedInterval(integerSequence(first, rest), first, size, values)
    case _: domainBuilt(buildDomainTree(size, values))

@unsafe
def domainSortedInterval(contiguous: Bool, first: U32, size: U32, values: +List<Val>) -> Domain:
  match contiguous:
    case True{}: IntDomain{first, size}
    case False{}: domainBuilt(buildDomainTree(size, values))

def domainBuilt(result: DomainBuilt) -> Domain:
  match result:
    case DomainBuilt{tree, _}: TreeDomain{tree}

# size is the exact remaining length; consume the sorted list in one pass.
@unsafe
def buildDomainTree(+size: U32, values: +List<Val>) -> DomainBuilt:
  match size:
    case 0: DomainBuilt{DomainEmpty{}, values}
    case _:
      +half = U32.div(size, 2)
      domainTreeLeft(buildDomainTree(half, values), U32.sub(U32.sub(size, half), 1), size)

@unsafe
def domainTreeLeft(result: DomainBuilt, right_size: U32, size: U32) -> DomainBuilt:
  match result:
    case DomainBuilt{left, value <> rest}:
      domainTreeRight(left, value, size, buildDomainTree(right_size, rest))
    case DomainBuilt{left, Nil{}}:
      DomainBuilt{left, Nil{}}

def domainTreeRight(left: DomainTree, value: Val, size: U32, result: DomainBuilt) -> DomainBuilt:
  match result:
    case DomainBuilt{right, rest}: DomainBuilt{DomainNode{left, value, right, size}, rest}

@unsafe
def domainTreeNth(tree: DomainTree, +index: U32) -> Val:
  match tree:
    case DomainEmpty{}: VE{"domain index out of bounds"}
    case DomainNode{+left, value, right, _}:
      +middle = domainTreeSize(left)
      domainTreeNthSide(U32.cmp(index, middle), left, value, right, U32.sub(U32.sub(index, middle), 1), index)

@unsafe
def domainTreeNthSide(order: Cmp, left: DomainTree, value: Val, right: DomainTree, after: U32, before: U32) -> Val:
  match order:
    case LT{}: domainTreeNth(left, before)
    case EQ{}: value
    case GT{}: domainTreeNth(right, after)

def domainNth(domain: Domain, +index: U32) -> Val:
  match domain:
    case IntDomain{first, size}:
      pickVal(U32.is_lt(index, size), VI{U32.add(first, index)}, VE{"domain index out of bounds"})
    case TreeDomain{tree}: domainTreeNth(tree, index)

@unsafe
def domainIndex(domain: Domain, value: Val) -> Maybe<U32>:
  match domain value:
    case IntDomain{first, size} VI{bits}:
      domainRangeIndex(U32.sub(bits, first), size)
    case TreeDomain{tree} v: domainTreeIndex(tree, v, 0)
    case _ _: None{}

def domainRangeRank(valid: Bool, index: U32) -> Maybe<U32>:
  match valid:
    case True{}: Some{index}
    case False{}: None{}

def domainRangeIndex(+index: U32, size: U32) -> Maybe<U32>:
  domainRangeRank(U32.is_lt(index, size), index)

@unsafe
def domainTreeIndex(tree: DomainTree, +value: Val, +offset: U32) -> Maybe<U32>:
  match tree:
    case DomainEmpty{}: None{}
    case DomainNode{+left, pivot, right, _}:
      domainTreeIndexSide(valueOrder(value, pivot), left, right, value, offset, U32.add(offset, domainTreeSize(left)))

@unsafe
def domainTreeIndexSide(order: Cmp, left: DomainTree, right: DomainTree, value: Val, before: U32, +middle: U32) -> Maybe<U32>:
  match order:
    case LT{}: domainTreeIndex(left, value, before)
    case EQ{}: Some{middle}
    case GT{}: domainTreeIndex(right, value, U32.inc(middle))

def nthCell(xs: +List<Cell>, n: Nat) -> Cell:
  match xs n:
    case Nil{} _:
      Cell{0, None{}, TreeDomain{DomainEmpty{}}, 1}
    case h <> t 0n:
      h
    case _ <> t 1n+p:
      nthCell(t, p)

def nthNode(xs: +List<IExpr>, n: Nat) -> IExpr:
  match xs n:
    case Nil{} _:
      IBad{"missing IR expression"}
    case h <> t 0n:
      h
    case _ <> t 1n+p:
      nthNode(t, p)

@unsafe
def decode(cells: +List<Cell>, id: U32) -> +List<Val>:
  match cells:
    case Nil{}:
      Nil{}
    case h <> t:
      decodeCell(h, t, id)

def decodeCell(c: Cell, rest: +List<Cell>, +id: U32) -> +List<Val>:
  match c:
    case Cell{owner, key, +dom, place}:
      +size = domainSize(dom)
      domainNth(dom, U32.mod(U32.div(id, place), size)) <> decode(rest, id)

# -------------------------------------------------------------------- encoding

@unsafe
def encode(cells: +List<Cell>, vals: +List<Val>) -> Maybe<U32>:
  encodeGo(cells, vals, 0)

@unsafe
def encodeGo(cells: +List<Cell>, vals: +List<Val>, acc: U32) -> Maybe<U32>:
  match cells:
    case Nil{}:
      encodeTail(vals, acc)
    case h <> t:
      encodeHead(h, t, vals, acc)

def encodeTail(vals: +List<Val>, acc: U32) -> Maybe<U32>:
  match vals:
    case Nil{}:
      Some{acc}
    case _:
      None{}

@unsafe
def encodeHead(c: Cell, rest: +List<Cell>, vals: +List<Val>, acc: U32) -> Maybe<U32>:
  match vals:
    case Nil{}:
      None{}
    case v <> vs:
      encodeCell(c, v, rest, vs, acc)

def encodeDigit(digit: Maybe<U32>, rest: +List<Cell>, vs: +List<Val>, acc: U32, place: U32) -> Maybe<U32>:
  match digit:
    case None{}: None{}
    case Some{rank}: encodeGo(rest, vs, U32.add(acc, U32.mul(place, rank)))

def encodeCell(c: Cell, v: Val, rest: +List<Cell>, vs: +List<Val>, +acc: U32) -> Maybe<U32>:
  match c:
    case Cell{owner, key, dom, +place}:
      encodeDigit(domainIndex(dom, v), rest, vs, acc, place)

# ----------------------------------------------------------- finite set values

def emptySet() -> Val:
  VSet{Nil{}, TreeDomain{DomainEmpty{}}}

@unsafe
def setInputErrorValue(values: +List<Val>) -> Maybe<String>:
  match values:
    case Nil{}: None{}
    case VE{message} <> _: Some{message}
    case _ <> rest: setInputErrorValue(rest)

@unsafe
def makeSet(+values: +List<Val>) -> Val:
  setInputError(setInputErrorValue(values), values)

@unsafe
def setInputError(error: Maybe<String>, +values: +List<Val>) -> Val:
  match error:
    case Some{message}: VE{message}
    case None{}: setUnique(List.sort(~Val, ~valueLe, values), None{}, Nil{}, 0)

@unsafe
def setSorted(+items: +List<Val>, +size: U32) -> Val:
  setSortedCapacity(U32.is_le(size, Core.state_capacity()), items, size)

@unsafe
def setSortedCapacity(fits: Bool, +items: +List<Val>, size: U32) -> Val:
  match fits:
    case False{}: VE{"finite set exceeds supported allocation capacity"}
    case True{}: VSet{items, domainSorted(items, size)}

@unsafe
def setUnique(values: +List<Val>, previous: +Maybe<Val>, reversed: +List<Val>, +size: U32) -> Val:
  match values:
    case Nil{}: setSorted(List.reverse(&2, Val, reversed), size)
    case +value <> rest: setUniquePrevious(previous, value, rest, reversed, size)

@unsafe
def setUniquePrevious(previous: +Maybe<Val>, +value: Val, rest: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match previous:
    case None{}: setUnique(rest, Some{value}, value <> reversed, U32.inc(size))
    case Some{old}: setUniqueDuplicate(veq(old, value), value, rest, reversed, size)

@unsafe
def setUniqueDuplicate(duplicate: Bool, +value: Val, rest: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match duplicate:
    case True{}: setUnique(rest, Some{value}, reversed, size)
    case False{}: setUnique(rest, Some{value}, value <> reversed, U32.inc(size))

@unsafe
def setAppend(values: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match values:
    case Nil{}: setSorted(List.reverse(&2, Val, reversed), size)
    case value <> rest: setAppend(rest, value <> reversed, U32.inc(size))

# Sorted merge modes: 1 union, 2 intersection, 3 difference.
@unsafe
def setMerge(mode: U32, left: +List<Val>, right: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match left right:
    case Nil{} ys: setMergeLeftEmpty(mode, ys, reversed, size)
    case xs Nil{}: setMergeRightEmpty(mode, xs, reversed, size)
    case +x <> xs +y <> ys:
      setMergeOrder(valueOrder(x, y), mode, x, xs, y, ys, reversed, size)

@unsafe
def setMergeLeftEmpty(mode: U32, right: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match mode:
    case 1: setAppend(right, reversed, size)
    case _: setSorted(List.reverse(&2, Val, reversed), size)

@unsafe
def setMergeRightEmpty(mode: U32, left: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match mode:
    case 2: setSorted(List.reverse(&2, Val, reversed), size)
    case _: setAppend(left, reversed, size)

@unsafe
def setMergeOrder(order: Cmp, mode: U32, +x: Val, +xs: +List<Val>, +y: Val, +ys: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match order:
    case LT{}: setMergeLess(mode, x, xs, y <> ys, reversed, size)
    case EQ{}: setMergeEqual(mode, x, xs, ys, reversed, size)
    case GT{}: setMergeGreater(mode, x <> xs, y, ys, reversed, size)

@unsafe
def setMergeLess(mode: U32, value: Val, left: +List<Val>, right: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match mode:
    case 2: setMerge(mode, left, right, reversed, size)
    case _: setMerge(mode, left, right, value <> reversed, U32.inc(size))

@unsafe
def setMergeEqual(mode: U32, value: Val, left: +List<Val>, right: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match mode:
    case 3: setMerge(mode, left, right, reversed, size)
    case _: setMerge(mode, left, right, value <> reversed, U32.inc(size))

@unsafe
def setMergeGreater(mode: U32, left: +List<Val>, value: Val, right: +List<Val>, reversed: +List<Val>, size: U32) -> Val:
  match mode:
    case 1: setMerge(mode, left, right, value <> reversed, U32.inc(size))
    case _: setMerge(mode, left, right, reversed, size)

@unsafe
def setSubset(left: +List<Val>, right: +List<Val>) -> Bool:
  match left right:
    case Nil{} _: True{}
    case _ Nil{}: False{}
    case +x <> xs y <> ys: setSubsetOrder(valueOrder(x, y), x, xs, ys)

@unsafe
def setSubsetOrder(order: Cmp, x: Val, xs: +List<Val>, ys: +List<Val>) -> Bool:
  match order:
    case LT{}: False{}
    case EQ{}: setSubset(xs, ys)
    case GT{}: setSubset(x <> xs, ys)

def memberResult(found: Maybe<U32>, negate: Bool) -> Val:
  match found:
    case None{}: VB{negate}
    case Some{_}: VB{Bool.not(negate)}

@unsafe
def valueMember(value: Val, set: Val, negate: Bool) -> Val:
  match value set:
    case VE{message} _: VE{message}
    case _ VE{message}: VE{message}
    case item VSet{_, index}: memberResult(domainIndex(index, item), negate)
    case _ _: VE{"membership requires a finite set"}

@unsafe
def valueRange(left: Val, right: Val) -> Val:
  match left right:
    case VE{message} _: VE{message}
    case _ VE{message}: VE{message}
    case VI{+lo} VI{+hi}: valueRangeOrdered(sle(lo, hi), lo, hi)
    case _ _: VE{"integer interval bounds must be integers"}

@unsafe
def valueRangeOrdered(ordered: Bool, +lo: U32, hi: U32) -> Val:
  match ordered:
    case False{}: emptySet()
    case True{}:
      +distance = U32.sub(hi, lo)
      valueRangeFits(U32.is_lt(distance, Core.state_capacity()), lo, U32.inc(distance))

@unsafe
def valueRangeFits(fits: Bool, +lo: U32, +size: U32) -> Val:
  match fits:
    case False{}: VE{"finite range exceeds supported allocation capacity"}
    case True{}: VSet{rangeItems(U32.to_nat(size), lo), IntDomain{lo, size}}

def rangeItems(fuel: Nat, +first: U32) -> +List<Val>:
  match fuel:
    case 0n: Nil{}
    case 1n+rest: VI{first} <> rangeItems(rest, U32.inc(first))

# Binary set opcodes: union/intersection/difference, subset, in, notin, range.
@unsafe
def setBinary(op: U32, left: Val, right: Val) -> Val:
  match op:
    case 5: valueMember(left, right, False{})
    case 6: valueMember(left, right, True{})
    case 7: valueRange(left, right)
    case _: setBinaryValues(op, left, right)

@unsafe
def setBinaryValues(op: U32, left: Val, right: Val) -> Val:
  match left right:
    case VE{message} _: VE{message}
    case _ VE{message}: VE{message}
    case VSet{xs, xi} VSet{ys, yi}: setBinaryIndexed(op, xs, xi, ys, yi)
    case _ _: VE{"set operators require finite set operands"}

@unsafe
def setBinaryIndexed(op: U32, +left: +List<Val>, left_index: Domain, +right: +List<Val>, right_index: Domain) -> Val:
  match op left right:
    case 1 Nil{} _: VSet{right, right_index}
    case 1 _ Nil{}: VSet{left, left_index}
    case 2 Nil{} _: VSet{Nil{}, left_index}
    case 2 _ Nil{}: VSet{Nil{}, right_index}
    case 3 Nil{} _: VSet{Nil{}, left_index}
    case 3 _ Nil{}: VSet{left, left_index}
    case 4 _ _: setSubsetFits(U32.is_le(domainSize(left_index), domainSize(right_index)), left, right)
    case _ _ _: setBinaryLists(op, left, right)

def setSubsetFits(fits: Bool, left: +List<Val>, right: +List<Val>) -> Val:
  match fits:
    case False{}: VB{False{}}
    case True{}: VB{setSubset(left, right)}

@unsafe
def setBinaryLists(op: U32, left: +List<Val>, right: +List<Val>) -> Val:
  match op:
    case 1: setMerge(1, left, right, Nil{}, 0)
    case 2: setMerge(2, left, right, Nil{}, 0)
    case 3: setMerge(3, left, right, Nil{}, 0)
    case 4: VB{setSubset(left, right)}
    case _: VE{"unknown lowered set operator"}

@unsafe
def powerCount(+remaining: U32, +size: U32) -> Maybe<U32>:
  match remaining:
    case 0: Some{size}
    case _: powerCountFits(U32.is_le(size, U32.div(Core.state_capacity(), 2)), U32.sub(remaining, 1), U32.add(size, size))

@unsafe
def powerCountFits(fits: Bool, remaining: U32, size: U32) -> Maybe<U32>:
  match fits:
    case False{}: None{}
    case True{}: powerCount(remaining, size)

@unsafe
def powerSet(allowed: Maybe<U32>, values: +List<Val>) -> Val:
  match allowed:
    case None{}: VE{"power set exceeds supported allocation capacity"}
    case Some{_}: powerItems(values)

@unsafe
def powerItems(values: +List<Val>) -> Val:
  match values:
    case Nil{}: setSorted([emptySet()], 1)
    case value <> rest: powerExtend(value, powerItems(rest))

@unsafe
def powerExtend(+value: Val, subsets: Val) -> Val:
  match subsets:
    case VE{message}: VE{message}
    case VSet{+items, _}: powerAdded(items, powerAdd(value, items))
    case _: VE{"invalid lowered power-set value"}

@unsafe
def powerAdd(+value: Val, subsets: +List<Val>) -> +List<Val>:
  match subsets:
    case Nil{}: Nil{}
    case VSet{items, index} <> rest:
      setSorted(value <> items, U32.inc(domainSize(index))) <> powerAdd(value, rest)
    case _ <> rest: VE{"invalid lowered power-set element"} <> powerAdd(value, rest)

@unsafe
def powerAdded(original: +List<Val>, added: +List<Val>) -> Val:
  setMerge(1, original, added, Nil{}, 0)

@unsafe
def unionItems(values: +List<Val>, acc: Val) -> Val:
  match values:
    case Nil{}: acc
    case value <> rest: unionAfter(setBinary(1, acc, value), rest)

@unsafe
def unionAfter(acc: Val, rest: +List<Val>) -> Val:
  match acc:
    case VE{message}: VE{message}
    case _: unionItems(rest, acc)

@unsafe
def setUnary(op: U32, value: Val) -> Val:
  match value:
    case VE{message}: VE{message}
    case VSet{+items, +index}: setUnaryValue(op, items, index)
    case _: VE{"finite set operand required"}

@unsafe
def setUnaryValue(op: U32, items: +List<Val>, +index: Domain) -> Val:
  match op:
    case 1: VI{domainSize(index)}
    case 2: powerSet(powerCount(domainSize(index), 1), items)
    case 3: unionItems(items, emptySet())
    case 4: VB{True{}}
    case _: VE{"unknown lowered unary set operator"}

# --------------------------------------------------------------- JSON objects

@unsafe
def catText(xs: +List<String>, ys: +List<String>) -> +List<String>:
  match xs:
    case Nil{}:
      ys
    case h <> t:
      h <> catText(t, ys)

def catEdges(xs: +List<Core.Edge>, ys: +List<Core.Edge>) -> +List<Core.Edge>:
  match xs:
    case Nil{}:
      ys
    case h <> t:
      h <> catEdges(t, ys)

def catVals(xs: +List<Val>, ys: +List<Val>) -> +List<Val>:
  match xs:
    case Nil{}:
      ys
    case h <> t:
      h <> catVals(t, ys)

@unsafe
def varParts(+cells: +List<Cell>, +vals: +List<Val>, vars: +List<Var>) -> +List<String>:
  match vars:
    case Nil{}:
      Nil{}
    case h <> t:
      catText(partOf(h, cells, vals), varParts(cells, vals, t))

@unsafe
def partOf(v: Var, cells: +List<Cell>, vals: +List<Val>) -> +List<String>:
  match v:
    case Var{name, start, count, function}:
      match function:
        case True{}:
          [Core.json_quote(name) ++ ":" ++ showFun(cells, vals, start, count)]
        case False{}:
          [Core.json_quote(name) ++ ":" ++ showVal(nthVal(vals, U32.to_nat(start), VE{"missing scalar cell"}))]

@unsafe
def showFun(cells: +List<Cell>, vals: +List<Val>, start: U32, count: U32) -> String:
  "{\"function\":[" ++ String.join(funParts(cells, vals, start, U32.to_nat(count)), ",") ++ "]}"

@unsafe
def funParts(+cells: +List<Cell>, +vals: +List<Val>, +start: U32, n: Nat) -> +List<String>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      catText(keyPart(nthCell(cells, U32.to_nat(start)), vals, start), funParts(cells, vals, U32.inc(start), p))

def keyPart(c: Cell, vals: +List<Val>, start: U32) -> +List<String>:
  match c:
    case Cell{owner, key, dom, place}:
      match key:
        case Some{k}:
          ["[" ++ showVal(k) ++ "," ++ showVal(nthVal(vals, U32.to_nat(start), VE{"missing function cell"})) ++ "]"]
        case None{}:
          Nil{}

def showValues(cells: +List<Cell>, vars: +List<Var>, vals: +List<Val>) -> String:
  "{" ++ String.join(varParts(cells, vals, vars), ",") ++ "}"

def showState(+cells: +List<Cell>, +vars: +List<Var>, state: U32) -> String:
  showValues(cells, vars, decode(cells, state))

# -------------------------------------------------------- expression evaluation

@unsafe
def eval(+nodes: +List<IExpr>, +env: EvalEnv, id: U32) -> Val:
  evalNode(nodes, env, nthNode(nodes, U32.to_nat(id)))

@unsafe
def evalNode(+nodes: +List<IExpr>, +env: EvalEnv, n: IExpr) -> Val:
  match n:
    case IInt{v}:
      VI{v}
    case IBool{b}:
      VB{b}
    case IStr{s}:
      VS{s}
    case ISlot{slot}:
      stateValue(env, slot)
    case IValue{value}:
      value
    case ILocal{slot}:
      localValue(env, slot)
    case ISet{items}:
      makeSet(evalSetItems(nodes, env, items))
    case ISetBinary{op, left, right}:
      setBinary(op, eval(nodes, env, left), eval(nodes, env, right))
    case ISetUnary{op, value}:
      setUnary(op, eval(nodes, env, value))
    case IBounded{op, slot, domain, body}:
      boundedValue(nodes, env, op, slot, body, eval(nodes, env, domain))
    case IFunction{keys, values}:
      functionSorted(keys, evalSetItems(nodes, env, values))
    case IProduct{keys, domains}:
      functionProduct(keys, evalSetItems(nodes, env, domains))
    case ILookup{function, key}:
      functionLookup(eval(nodes, env, function), eval(nodes, env, key))
    case IUpdate{function, key, slot, value}:
      functionUpdate(nodes, env, slot, value, eval(nodes, env, function), eval(nodes, env, key))
    case IDomain{function}:
      functionDomain(eval(nodes, env, function))
    case ISequence{op, args}:
      sequenceCall(op, evalSetItems(nodes, env, args))
    case IAdd{l, r}:
      vadd(eval(nodes, env, l), eval(nodes, env, r))
    case ISub{l, r}:
      vsub(eval(nodes, env, l), eval(nodes, env, r))
    case INeg{a}:
      vneg(eval(nodes, env, a))
    case IEq{l, r}:
      veqb(eval(nodes, env, l), eval(nodes, env, r))
    case INe{l, r}:
      vneb(eval(nodes, env, l), eval(nodes, env, r))
    case ILt{l, r}:
      vlt(eval(nodes, env, l), eval(nodes, env, r))
    case ILe{l, r}:
      vle(eval(nodes, env, l), eval(nodes, env, r))
    case IGt{l, r}:
      vgt(eval(nodes, env, l), eval(nodes, env, r))
    case IGe{l, r}:
      vge(eval(nodes, env, l), eval(nodes, env, r))
    case IAnd{l, r}:
      andLeft(nodes, env, r, eval(nodes, env, l))
    case IOr{l, r}:
      orLeft(nodes, env, r, eval(nodes, env, l))
    case INot{a}:
      vnot(eval(nodes, env, a))
    case IImp{l, r}:
      implicationLeft(nodes, env, r, eval(nodes, env, l))
    case IIte{c, t, e}:
      conditionalValue(nodes, env, t, e, eval(nodes, env, c))
    case IBad{msg}:
      VE{msg}

def stateValue(env: EvalEnv, +slot: U32) -> Val:
  match env:
    case EvalEnv{state, _}:
      nthVal(state, U32.to_nat(slot), VE{"state slot " ++ U32.show(slot) ++ " is unavailable"})

@unsafe
def localValue(env: EvalEnv, slot: U32) -> Val:
  match env:
    case EvalEnv{_, locals}: localFind(locals, slot)

@unsafe
def localFind(locals: +List<Local>, +slot: U32) -> Val:
  match locals:
    case Nil{}: VE{"unbound numeric IR local"}
    case Local{+key, value} <> rest: localFound(U32.is_eq(key, slot), value, rest, slot)

@unsafe
def localFound(found: Bool, value: Val, rest: +List<Local>, slot: U32) -> Val:
  match found:
    case True{}: value
    case False{}: localFind(rest, slot)

def bindLocal(env: EvalEnv, slot: U32, value: Val) -> EvalEnv:
  match env:
    case EvalEnv{state, locals}: EvalEnv{state, Local{slot, value} <> locals}

@unsafe
def evalSetItems(+nodes: +List<IExpr>, +env: EvalEnv, items: +List<U32>) -> +List<Val>:
  match items:
    case Nil{}: Nil{}
    case id <> rest: eval(nodes, env, id) <> evalSetItems(nodes, env, rest)

# Bounded opcodes: forall, exists, filter, image. Locals are numeric IR slots.
@unsafe
def boundedValue(nodes: +List<IExpr>, env: EvalEnv, op: U32, slot: U32, body: U32, domain: Val) -> Val:
  match domain:
    case VE{message}: VE{message}
    case VSet{items, _}: boundedLoop(nodes, env, op, slot, body, items, Nil{})
    case _: VE{"bounded expression requires a finite set"}

@unsafe
def boundedLoop(+nodes: +List<IExpr>, +env: EvalEnv, op: U32, +slot: U32, +body: U32, remaining: +List<Val>, acc: +List<Val>) -> Val:
  match remaining:
    case Nil{}: boundedEmpty(op, acc)
    case +value <> rest:
      boundedStep(nodes, env, op, slot, body, value, rest, acc, eval(nodes, bindLocal(env, slot, value), body))

# Filtering preserves canonical input order; reverse/count once, without sorting.
@unsafe
def filteredFinish(reversed: +List<Val>, sorted: +List<Val>, size: U32) -> Val:
  match reversed:
    case Nil{}: setSorted(sorted, size)
    case value <> rest: filteredFinish(rest, value <> sorted, U32.inc(size))

def boundedEmpty(op: U32, acc: +List<Val>) -> Val:
  match op:
    case 1: VB{True{}}
    case 2: VB{False{}}
    case 3: filteredFinish(acc, Nil{}, 0)
    case 4: makeSet(acc)
    case _: VE{"unknown lowered bounded operator"}

@unsafe
def boundedStep(nodes: +List<IExpr>, env: EvalEnv, op: U32, slot: U32, body: U32, original: Val, rest: +List<Val>, acc: +List<Val>, value: Val) -> Val:
  match value:
    case VE{message}: VE{message}
    case _: boundedKind(nodes, env, op, slot, body, original, rest, acc, value)

@unsafe
def boundedKind(nodes: +List<IExpr>, env: EvalEnv, op: U32, slot: U32, body: U32, original: Val, rest: +List<Val>, acc: +List<Val>, value: Val) -> Val:
  match op:
    case 4: boundedLoop(nodes, env, op, slot, body, rest, value <> acc)
    case _: boundedBoolean(nodes, env, op, slot, body, original, rest, acc, value)

@unsafe
def boundedBoolean(nodes: +List<IExpr>, env: EvalEnv, op: U32, slot: U32, body: U32, original: Val, rest: +List<Val>, acc: +List<Val>, value: Val) -> Val:
  match value:
    case VB{truth}: boundedTruth(nodes, env, op, slot, body, original, rest, acc, truth)
    case _: VE{"bounded predicate must be boolean"}

@unsafe
def boundedTruth(nodes: +List<IExpr>, env: EvalEnv, op: U32, slot: U32, body: U32, original: Val, rest: +List<Val>, acc: +List<Val>, truth: Bool) -> Val:
  match truth:
    case False{}: boundedFalse(nodes, env, op, slot, body, original, rest, acc)
    case True{}: boundedTrue(nodes, env, op, slot, body, original, rest, acc)

@unsafe
def boundedFalse(nodes: +List<IExpr>, env: EvalEnv, op: U32, slot: U32, body: U32, original: Val, rest: +List<Val>, acc: +List<Val>) -> Val:
  match op:
    case 1: VB{False{}}
    case _: boundedLoop(nodes, env, op, slot, body, rest, acc)

@unsafe
def boundedTrue(nodes: +List<IExpr>, env: EvalEnv, op: U32, slot: U32, body: U32, original: Val, rest: +List<Val>, acc: +List<Val>) -> Val:
  match op:
    case 2: VB{True{}}
    case 3: boundedLoop(nodes, env, op, slot, body, rest, original <> acc)
    case _: boundedLoop(nodes, env, op, slot, body, rest, acc)

def booleanValue(value: Val) -> Val:
  match value:
    case VB{b}:
      VB{b}
    case VE{message}:
      VE{message}
    case _:
      VE{"expected a boolean operand"}

def andLeft(nodes: +List<IExpr>, env: EvalEnv, right: U32, left: Val) -> Val:
  match left:
    case VB{False{}}:
      VB{False{}}
    case VB{True{}}:
      booleanValue(eval(nodes, env, right))
    case VE{message}:
      VE{message}
    case _:
      VE{"expected booleans for /\\"}

def orLeft(nodes: +List<IExpr>, env: EvalEnv, right: U32, left: Val) -> Val:
  match left:
    case VB{True{}}:
      VB{True{}}
    case VB{False{}}:
      booleanValue(eval(nodes, env, right))
    case VE{message}:
      VE{message}
    case _:
      VE{"expected booleans for \\/"}

def implicationLeft(nodes: +List<IExpr>, env: EvalEnv, right: U32, left: Val) -> Val:
  match left:
    case VB{False{}}:
      VB{True{}}
    case VB{True{}}:
      booleanValue(eval(nodes, env, right))
    case VE{message}:
      VE{message}
    case _:
      VE{"expected booleans for implication"}

def conditionalBool(nodes: +List<IExpr>, env: EvalEnv, yes: U32, no: U32, condition: Bool) -> Val:
  match condition:
    case True{}:
      eval(nodes, env, yes)
    case False{}:
      eval(nodes, env, no)

def conditionalValue(nodes: +List<IExpr>, env: EvalEnv, yes: U32, no: U32, condition: Val) -> Val:
  match condition:
    case VB{b}:
      conditionalBool(nodes, env, yes, no, b)
    case VE{message}:
      VE{message}
    case _:
      VE{"expected a boolean condition"}

# ------------------------------------------------------------ action execution

@unsafe
def evalAssigns(+exprs: +List<IExpr>, +env: EvalEnv, assigns: +List<Assign>) -> +List<Val>:
  match assigns:
    case Nil{}:
      Nil{}
    case h <> t:
      catVals(evalOne(exprs, env, h), evalAssigns(exprs, env, t))

def evalOne(exprs: +List<IExpr>, env: EvalEnv, a: Assign) -> +List<Val>:
  match a:
    case Assign{slot, expr}:
      [eval(exprs, env, expr)]

@unsafe
def firstErr(xs: +List<Val>) -> Maybe<String>:
  match xs:
    case Nil{}:
      None{}
    case h <> t:
      firstErrHead(h, t)

def firstErrHead(v: Val, rest: +List<Val>) -> Maybe<String>:
  match v:
    case VE{msg}:
      Some{msg}
    case _:
      firstErr(rest)

@unsafe
def edgeOf(+cells: +List<Cell>, +vars: +List<Var>, +vals: +List<Val>) -> Core.Edge:
  edgeEncoded(cells, vars, vals, encode(cells, vals))

def edgeEncoded(cells: +List<Cell>, vars: +List<Var>, vals: +List<Val>, r: Maybe<U32>) -> Core.Edge:
  match r:
    case Some{id}:
      Core.To{id}
    case None{}:
      Core.Outside{showValues(cells, vars, vals)}

@unsafe
def edgeFromVals(cells: +List<Cell>, vars: +List<Var>, +vals: +List<Val>) -> Core.Edge:
  edgeChecked(cells, vars, vals, firstErr(vals))

def edgeChecked(cells: +List<Cell>, vars: +List<Var>, vals: +List<Val>, e: Maybe<String>) -> Core.Edge:
  match e:
    case Some{msg}:
      Core.Rejected{msg}
    case None{}:
      edgeOf(cells, vars, vals)

@unsafe
def actionGuard(exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, env: EvalEnv, assigns: +List<Assign>, g: Val) -> +List<Core.Edge>:
  match g:
    case VB{b}:
      actionBool(exprs, cells, vars, env, assigns, b)
    case VE{msg}:
      [Core.Rejected{msg}]
    case _:
      [Core.Rejected{"action guard is not boolean"}]

def actionBool(exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, env: EvalEnv, assigns: +List<Assign>, b: Bool) -> +List<Core.Edge>:
  match b:
    case True{}:
      [edgeFromVals(cells, vars, evalAssigns(exprs, env, assigns))]
    case False{}:
      Nil{}

@unsafe def actionOne(+exprs: +List<IExpr>, +cells: +List<Cell>, +vars: +List<Var>, +env: EvalEnv, a: Action) -> +List<Core.Edge>:
  match a:
    case Action{guard, assigns}:
      actionGuard(exprs, cells, vars, env, assigns, eval(exprs, env, guard))
    case Relation{guard, choices, values, predicate}:
      relationGuard(exprs, cells, vars, env, choices, values, predicate, eval(exprs, env, guard))

def runActions(+exprs: +List<IExpr>, +cells: +List<Cell>, +vars: +List<Var>, +env: EvalEnv, actions: +List<Action>) -> +List<Core.Edge>:
  match actions:
    case Nil{}:
      Nil{}
    case h <> t:
      catEdges(actionOne(exprs, cells, vars, env, h), runActions(exprs, cells, vars, env, t))

def nextEdges(exprs: +List<IExpr>, +cells: +List<Cell>, vars: +List<Var>, actions: +List<Action>, state: U32) -> +List<Core.Edge>:
  runActions(exprs, cells, vars, EvalEnv{decode(cells, state), Nil{}}, actions)

# ------------------------------------------------------------------ initial

def caseEdges(exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, c: Case) -> +List<Core.Edge>:
  match c:
    case Case{assigns}:
      [edgeFromVals(cells, vars, evalAssigns(exprs, EvalEnv{Nil{}, Nil{}}, assigns))]

def initialEdges(+exprs: +List<IExpr>, +cells: +List<Cell>, +vars: +List<Var>, cases: +List<Case>) -> +List<Core.Edge>:
  match cases:
    case Nil{}:
      Nil{}
    case h <> t:
      catEdges(caseEdges(exprs, cells, vars, h), initialEdges(exprs, cells, vars, t))

# ---------------------------------------------------------------- invariants

@unsafe
def invValue(exprs: +List<IExpr>, env: EvalEnv, name: String, rest: +List<Named>, v: Val) -> Core.InvariantCheck:
  match v:
    case VB{b}:
      invBool(exprs, env, name, rest, b)
    case VE{msg}:
      Core.InvariantError{"invariant " ++ name ++ ": " ++ msg}
    case _:
      Core.InvariantError{"invariant " ++ name ++ ": result is not boolean"}

@unsafe
def invBool(exprs: +List<IExpr>, env: EvalEnv, name: String, rest: +List<Named>, b: Bool) -> Core.InvariantCheck:
  match b:
    case True{}:
      invGo(exprs, env, rest)
    case False{}:
      Core.InvariantViolation{name}

def invOne(+exprs: +List<IExpr>, +env: EvalEnv, n: Named, rest: +List<Named>) -> Core.InvariantCheck:
  match n:
    case Named{name, expr}:
      invValue(exprs, env, name, rest, eval(exprs, env, expr))

def invGo(exprs: +List<IExpr>, env: EvalEnv, invs: +List<Named>) -> Core.InvariantCheck:
  match invs:
    case Nil{}:
      Core.InvariantHolds{}
    case h <> t:
      invOne(exprs, env, h, t)

@unsafe
def checkInvariants(exprs: +List<IExpr>, cells: +List<Cell>, invs: +List<Named>, state: U32) -> Core.InvariantCheck:
  invGo(exprs, EvalEnv{decode(cells, state), Nil{}}, invs)

# ----------------------------------------------------- finite function values
def functionSize(tree: FunctionTree) -> U32:
  match tree:
    case FunctionEmpty{}: 0
    case FunctionNode{_, _, _, _, size, _}: size

def functionHeight(tree: FunctionTree) -> U32:
  match tree:
    case FunctionEmpty{}: 0
    case FunctionNode{_, _, _, _, _, height}: height

def functionMax(greater: Bool, a: U32, b: U32) -> U32:
  match greater:
    case True{}: a
    case False{}: b

def functionNode(+left: FunctionTree, key: Val, value: Val, +right: FunctionTree) -> FunctionTree:
  +lh = functionHeight(left)
  +rh = functionHeight(right)
  FunctionNode{left, key, value, right, U32.inc(U32.add(functionSize(left), functionSize(right))), U32.inc(functionMax(U32.is_ge(lh, rh), lh, rh))}

def functionEqual(order: Cmp) -> Bool:
  match order:
    case EQ{}: True{}
    case _: False{}

@unsafe
def functionStack(+tree: FunctionTree, rest: +List<FunctionTree>) -> +List<FunctionTree>:
  match tree:
    case FunctionEmpty{}: rest
    case FunctionNode{left, _, _, _, _, _}:
      functionStack(left, tree <> rest)

@unsafe
def functionOrder(+left: FunctionTree, +right: FunctionTree) -> Cmp:
  functionOrderSize(U32.cmp(functionSize(left), functionSize(right)), left, right)

@unsafe
def functionOrderSize(order: Cmp, left: FunctionTree, right: FunctionTree) -> Cmp:
  match order:
    case EQ{}: functionOrderStacks(functionStack(left, Nil{}), functionStack(right, Nil{}))
    case _: order

@unsafe
def functionOrderStacks(xs: +List<FunctionTree>, ys: +List<FunctionTree>) -> Cmp:
  match xs ys:
    case Nil{} Nil{}: EQ{}
    case FunctionNode{_, +key, +value, right, _, _} <> rest FunctionNode{_, +other, +actual, next, _, _} <> more:
      functionOrderEntry(valueOrder(key, other), value, actual, right, rest, next, more)
    case Nil{} _: LT{}
    case _ Nil{}: GT{}
    case _ _: EQ{}

@unsafe
def functionOrderEntry(order: Cmp, value: Val, actual: Val, right: FunctionTree, rest: +List<FunctionTree>, next: FunctionTree, more: +List<FunctionTree>) -> Cmp:
  match order:
    case EQ{}: functionOrderValue(valueOrder(value, actual), right, rest, next, more)
    case _: order

@unsafe
def functionOrderValue(order: Cmp, right: FunctionTree, rest: +List<FunctionTree>, next: FunctionTree, more: +List<FunctionTree>) -> Cmp:
  match order:
    case EQ{}: functionOrderStacks(functionStack(right, rest), functionStack(next, more))
    case _: order

@unsafe
def functionKeys(tree: FunctionTree, acc: +List<Val>) -> +List<Val>:
  match tree:
    case FunctionEmpty{}: acc
    case FunctionNode{left, key, _, right, _, _}: functionKeys(left, key <> functionKeys(right, acc))

@unsafe
def functionValues(tree: FunctionTree, acc: +List<Val>) -> +List<Val>:
  match tree:
    case FunctionEmpty{}: acc
    case FunctionNode{left, _, value, right, _, _}: functionValues(left, value <> functionValues(right, acc))

@unsafe
def functionJson(tree: FunctionTree, acc: +List<String>) -> +List<String>:
  match tree:
    case FunctionEmpty{}: acc
    case FunctionNode{left, key, value, right, _, _}:
      functionJson(left, ("[" ++ showVal(key) ++ "," ++ showVal(value) ++ "]") <> functionJson(right, acc))

@unsafe
def functionFind(tree: FunctionTree, +key: Val) -> Maybe<Val>:
  match tree:
    case FunctionEmpty{}: None{}
    case FunctionNode{left, +other, value, right, _, _}:
      functionFindOrder(valueOrder(key, other), left, value, right, key)

@unsafe
def functionFindOrder(order: Cmp, left: FunctionTree, value: Val, right: FunctionTree, key: Val) -> Maybe<Val>:
  match order:
    case EQ{}: Some{value}
    case LT{}: functionFind(left, key)
    case GT{}: functionFind(right, key)

def functionFound(value: Maybe<Val>) -> Val:
  match value:
    case Some{v}: v
    case None{}: VE{"function key is outside DOMAIN"}

@unsafe
def functionLookup(function: Val, key: Val) -> Val:
  match function key:
    case VE{message} _: VE{message}
    case _ VE{message}: VE{message}
    case VFun{tree, _} _: functionFound(functionFind(tree, key))
    case _ _: VE{"lookup requires a finite function"}

@unsafe
def functionReplace(tree: FunctionTree, +key: Val, value: Val) -> FunctionTree:
  match tree:
    case FunctionEmpty{}: FunctionEmpty{}
    case FunctionNode{left, +other, old, right, _, _}:
      functionReplaceOrder(valueOrder(key, other), left, other, old, right, key, value)

@unsafe
def functionReplaceOrder(order: Cmp, left: FunctionTree, other: Val, old: Val, right: FunctionTree, key: Val, value: Val) -> FunctionTree:
  match order:
    case EQ{}: functionNode(left, other, value, right)
    case LT{}: functionNode(functionReplace(left, key, value), other, old, right)
    case GT{}: functionNode(left, other, old, functionReplace(right, key, value))

@unsafe
def functionUpdate(nodes: +List<IExpr>, env: EvalEnv, slot: U32, body: U32, function: Val, +key: Val) -> Val:
  match function key:
    case VE{message} _: VE{message}
    case _ VE{message}: VE{message}
    case VFun{+tree, sequence} _:
      functionUpdateFound(functionFind(tree, key), nodes, env, slot, body, tree, sequence, key)
    case _ _: VE{"EXCEPT requires a finite function"}

@unsafe
def functionUpdateFound(found: Maybe<Val>, nodes: +List<IExpr>, env: EvalEnv, slot: U32, body: U32, tree: FunctionTree, sequence: Bool, key: Val) -> Val:
  match found:
    case None{}: VFun{tree, sequence}
    case Some{old}: functionUpdated(eval(nodes, bindLocal(env, slot, old), body), tree, sequence, key)

@unsafe
def functionUpdated(value: Val, tree: FunctionTree, sequence: Bool, key: Val) -> Val:
  match value:
    case VE{message}: VE{message}
    case _: VFun{functionReplace(tree, key, value), sequence}

@unsafe
def functionDomain(function: Val) -> Val:
  match function:
    case VE{message}: VE{message}
    case VFun{+tree, _}: setSorted(functionKeys(tree, Nil{}), functionSize(tree))
    case _: VE{"DOMAIN requires a finite function"}

@unsafe
def functionSequenceKeys(keys: +List<Val>, +next: U32) -> Bool:
  match keys:
    case Nil{}: True{}
    case VI{+key} <> rest: functionSequenceKey(U32.is_eq(key, next), rest, U32.inc(next))
    case _: False{}

@unsafe
def functionSequenceKey(equal: Bool, keys: +List<Val>, next: U32) -> Bool:
  match equal:
    case True{}: functionSequenceKeys(keys, next)
    case False{}: False{}

@unsafe
def functionSorted(+keys: +List<Val>, +values: +List<Val>) -> Val:
  functionSortedValid(firstErr(values), keys, values)

@unsafe
def functionSortedValid(error: Maybe<String>, +keys: +List<Val>, +values: +List<Val>) -> Val:
  match error:
    case Some{message}: VE{message}
    case None{}: functionSortedSizes(vlen(keys), vlen(values), keys, values)

@unsafe
def functionSortedSizes(+size: U32, actual: U32, keys: +List<Val>, values: +List<Val>) -> Val:
  functionSortedLength(U32.is_eq(size, actual), size, keys, values)

@unsafe
def functionSortedLength(valid: Bool, +size: U32, keys: +List<Val>, values: +List<Val>) -> Val:
  match valid:
    case False{}: VE{"malformed lowered finite function"}
    case True{}: functionSortedCapacity(U32.is_le(size, Core.state_capacity()), size, keys, values)

@unsafe
def functionSortedCapacity(fits: Bool, size: U32, +keys: +List<Val>, values: +List<Val>) -> Val:
  match fits:
    case False{}: VE{"finite function exceeds supported allocation capacity"}
    case True{}: functionBuiltValue(buildFunctionTree(size, keys, values), functionSequenceKeys(keys, 1))

def functionBuiltValue(built: FunctionBuilt, sequence: Bool) -> Val:
  match built:
    case FunctionBuilt{tree, Nil{}, Nil{}}: VFun{tree, sequence}
    case _: VE{"malformed lowered finite function"}

@unsafe
def buildFunctionTree(+size: U32, keys: +List<Val>, values: +List<Val>) -> FunctionBuilt:
  match size:
    case 0: FunctionBuilt{FunctionEmpty{}, keys, values}
    case _: buildFunctionLeft(buildFunctionTree(U32.div(size, 2), keys, values), U32.sub(U32.sub(size, 1), U32.div(size, 2)))

@unsafe
def buildFunctionLeft(built: FunctionBuilt, right_size: U32) -> FunctionBuilt:
  match built:
    case FunctionBuilt{left, key <> keys, value <> values}:
      buildFunctionRight(buildFunctionTree(right_size, keys, values), left, key, value)
    case _: built

def buildFunctionRight(built: FunctionBuilt, left: FunctionTree, key: Val, value: Val) -> FunctionBuilt:
  match built:
    case FunctionBuilt{right, keys, values}: FunctionBuilt{functionNode(left, key, value, right), keys, values}

# Appending a greatest key to an AVL tree needs only right-heavy rotations.
@unsafe
def functionAppendTree(tree: FunctionTree, key: Val, value: Val) -> FunctionTree:
  match tree:
    case FunctionEmpty{}: functionNode(FunctionEmpty{}, key, value, FunctionEmpty{})
    case FunctionNode{left, other, old, right, _, _}:
      functionAppendBalanced(left, other, old, functionAppendTree(right, key, value))

@unsafe
def functionAppendBalanced(+left: FunctionTree, key: Val, value: Val, +right: FunctionTree) -> FunctionTree:
  functionAppendBalance(U32.is_gt(functionHeight(right), U32.inc(functionHeight(left))), left, key, value, right)

def functionAppendBalance(rotate: Bool, left: FunctionTree, key: Val, value: Val, right: FunctionTree) -> FunctionTree:
  match rotate right:
    case True{} FunctionNode{middle, other, old, last, _, _}:
      functionNode(functionNode(left, key, value, middle), other, old, last)
    case _ _: functionNode(left, key, value, right)

@unsafe
def sequenceValue(+values: +List<Val>) -> Val:
  functionSorted(rangeItems(U32.to_nat(vlen(values)), 1), values)

def emptyFunction() -> Val:
  VFun{FunctionEmpty{}, True{}}

@unsafe
def sequenceCall(op: U32, +args: +List<Val>) -> Val:
  sequenceChecked(firstErr(args), op, args)

@unsafe
def sequenceChecked(error: Maybe<String>, op: U32, args: +List<Val>) -> Val:
  match error:
    case Some{message}: VE{message}
    case None{}: sequenceArgs(op, args)

@unsafe
def sequenceArgs(op: U32, args: +List<Val>) -> Val:
  match op:
    case 1: sequenceLengthArg(args)
    case 2: sequenceHeadArg(args)
    case 3: sequenceTailArg(args)
    case 4: sequenceAppendArg(args)
    case 5: sequenceConcatArgs(args)
    case 6: sequenceRangeArgs(args)
    case _: VE{"unknown lowered sequence operator"}

def sequenceLengthArg(args: +List<Val>) -> Val:
  match args:
    case VFun{tree, True{}} <> Nil{}: VI{functionSize(tree)}
    case _: VE{"Len requires a tuple"}

@unsafe
def sequenceHeadArg(args: +List<Val>) -> Val:
  match args:
    case VFun{tree, True{}} <> Nil{}: functionFound(functionFind(tree, VI{1}))
    case _: VE{"Head requires a tuple"}

@unsafe
def sequenceTailArg(args: +List<Val>) -> Val:
  match args:
    case VFun{+tree, True{}} <> Nil{}: sequenceTail(functionSize(tree), tree)
    case _: VE{"Tail requires a tuple"}

@unsafe
def sequenceAppendArg(args: +List<Val>) -> Val:
  match args:
    case VFun{+tree, True{}} <> value <> Nil{}: sequenceAppend(U32.is_lt(functionSize(tree), Core.state_capacity()), tree, value)
    case _: VE{"Append requires a tuple and a value"}

@unsafe
def sequenceConcatArgs(args: +List<Val>) -> Val:
  match args:
    case VFun{left, True{}} <> VFun{right, True{}} <> Nil{}: sequenceConcat(left, right)
    case _: VE{"concatenation requires two tuples"}

@unsafe
def sequenceRangeArgs(args: +List<Val>) -> Val:
  match args:
    case VFun{tree, True{}} <> VI{first} <> VI{last} <> Nil{}: sequenceRange(tree, first, last)
    case _: VE{"SubSeq requires a tuple and integer endpoints"}

@unsafe
def sequenceTail(size: U32, tree: FunctionTree) -> Val:
  match size:
    case 0: VE{"Tail requires a nonempty sequence"}
    case _: sequenceValue(sequenceDrop(functionValues(tree, Nil{}), 1n))

def sequenceAppend(fits: Bool, +tree: FunctionTree, value: Val) -> Val:
  match fits:
    case False{}: VE{"sequence exceeds supported allocation capacity"}
    case True{}: VFun{functionAppendTree(tree, VI{U32.inc(functionSize(tree))}, value), True{}}

@unsafe
def sequenceConcat(+left: FunctionTree, +right: FunctionTree) -> Val:
  sequenceConcatSize(functionSize(left), functionSize(right), left, right)

@unsafe
def sequenceConcatSize(+left_size: U32, +right_size: U32, left: FunctionTree, right: FunctionTree) -> Val:
  sequenceConcatEmpty(U32.is_eq(left_size, 0), U32.is_eq(right_size, 0), left_size, right_size, left, right)

@unsafe
def sequenceConcatEmpty(left_empty: Bool, right_empty: Bool, left_size: U32, right_size: U32, left: FunctionTree, right: FunctionTree) -> Val:
  match left_empty right_empty:
    case True{} _: VFun{right, True{}}
    case _ True{}: VFun{left, True{}}
    case _ _: sequenceConcatFits(U32.is_le(left_size, U32.sub(Core.state_capacity(), right_size)), left, right)

@unsafe
def sequenceConcatFits(fits: Bool, left: FunctionTree, right: FunctionTree) -> Val:
  match fits:
    case False{}: VE{"sequence exceeds supported allocation capacity"}
    case True{}: sequenceValue(functionValues(left, functionValues(right, Nil{})))

def sequenceDrop(values: +List<Val>, count: Nat) -> +List<Val>:
  match values count:
    case _ 0n: values
    case _ <> more 1n+rest: sequenceDrop(more, rest)
    case Nil{} _: Nil{}

def sequenceTake(values: +List<Val>, count: Nat) -> +List<Val>:
  match values count:
    case _ 0n: Nil{}
    case value <> more 1n+rest: value <> sequenceTake(more, rest)
    case Nil{} _: Nil{}

@unsafe
def sequenceRange(+tree: FunctionTree, +first: U32, +last: U32) -> Val:
  sequenceRangeEmpty(sgt(first, last), tree, first, last)

@unsafe
def sequenceRangeEmpty(empty: Bool, +tree: FunctionTree, +first: U32, +last: U32) -> Val:
  match empty:
    case True{}: emptyFunction()
    case False{}: sequenceRangeValid(Bool.and(sge(first, 1), sle(last, functionSize(tree))), tree, first, last)

@unsafe
def sequenceRangeValid(valid: Bool, +tree: FunctionTree, +first: U32, +last: U32) -> Val:
  match valid:
    case False{}: VE{"SubSeq endpoint is outside DOMAIN"}
    case True{}: sequenceRangeWhole(Bool.and(U32.is_eq(first, 1), U32.is_eq(last, functionSize(tree))), tree, first, last)

@unsafe
def sequenceRangeWhole(whole: Bool, tree: FunctionTree, +first: U32, +last: U32) -> Val:
  match whole:
    case True{}: VFun{tree, True{}}
    case False{}: sequenceValue(sequenceTake(sequenceDrop(functionValues(tree, Nil{}), U32.to_nat(U32.sub(first, 1))), U32.to_nat(U32.inc(U32.sub(last, first)))))

# Product cardinalities are checked before materializing any function rows.
@unsafe
def productCount(domains: +List<Val>, +count: U32) -> Val:
  match domains:
    case Nil{}: productCountEnd(U32.is_le(count, Core.state_capacity()), count)
    case VE{message} <> _: VE{message}
    case VSet{_, +index} <> rest: productCountSize(domainSize(index), rest, count)
    case _: VE{"finite function ranges must be finite sets"}

def productCountEnd(fits: Bool, count: U32) -> Val:
  match fits:
    case True{}: VI{count}
    case False{}: VE{"finite function set exceeds supported allocation capacity"}

@unsafe
def productCountSize(+size: U32, rest: +List<Val>, +count: U32) -> Val:
  match size:
    case 0: productCount(rest, 0)
    case _: productCountFits(U32.is_le(count, U32.div(Core.state_capacity(), size)), rest, count, size)

@unsafe
def productCountFits(fits: Bool, rest: +List<Val>, count: U32, size: U32) -> Val:
  match fits:
    case False{}: productCount(rest, U32.inc(Core.state_capacity()))
    case True{}: productCount(rest, U32.mul(count, size))

@unsafe
def functionProduct(keys: +List<Val>, +domains: +List<Val>) -> Val:
  functionProductCount(productCount(domains, 1), keys, domains)

@unsafe
def functionProductCount(count: Val, keys: +List<Val>, domains: +List<Val>) -> Val:
  match count:
    case VE{message}: VE{message}
    case VI{+size}: functionProductEmpty(U32.is_eq(size, 0), size, keys, domains)
    case _: VE{"malformed finite function product"}

@unsafe
def functionProductEmpty(empty: Bool, size: U32, keys: +List<Val>, domains: +List<Val>) -> Val:
  match empty:
    case True{}: emptySet()
    case False{}: setSorted(productFunctions(keys, productRows(domains)), size)

@unsafe
def productRows(domains: +List<Val>) -> +List<+List<Val>>:
  match domains:
    case Nil{}: [Nil{}]
    case VSet{items, _} <> rest: productColumn(items, productRows(rest))
    case _: Nil{}

@unsafe
def productColumn(values: +List<Val>, +rows: +List<+List<Val>>) -> +List<+List<Val>>:
  match values:
    case Nil{}: Nil{}
    case value <> rest: productPrefix(value, rows, productColumn(rest, rows))

@unsafe
def productPrefix(+value: Val, rows: +List<+List<Val>>, acc: +List<+List<Val>>) -> +List<+List<Val>>:
  match rows:
    case Nil{}: acc
    case row <> rest: (value <> row) <> productPrefix(value, rest, acc)

@unsafe
def productFunctions(+keys: +List<Val>, rows: +List<+List<Val>>) -> +List<Val>:
  match rows:
    case Nil{}: Nil{}
    case row <> rest: functionSorted(keys, row) <> productFunctions(keys, rest)

# Finite relation choices bind numeric IR locals. TypeOK is consulted only after
# every relation predicate succeeds, when the complete successor is encoded.
@unsafe
def relationGuard(exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, env: EvalEnv, choices: +List<Choice>, outputs: +List<U32>, predicate: U32, guard: Val) -> +List<Core.Edge>:
  match guard:
    case VB{False{}}: Nil{}
    case VB{True{}}:
      relationEdges(relationLoop(exprs, cells, vars, env, choices, outputs, predicate, ChoiceResult{Nil{}, Core.state_capacity(), False{}}))
    case VE{message}: [Core.Rejected{message}]
    case _: [Core.Rejected{"relation guard is not boolean"}]

def relationEdges(result: ChoiceResult) -> +List<Core.Edge>:
  match result:
    case ChoiceResult{reversed, _, _}: List.reverse(&2, Core.Edge, reversed)

def relationFailure(result: ChoiceResult, message: String) -> ChoiceResult:
  match result:
    case ChoiceResult{edges, fuel, _}: ChoiceResult{Core.Rejected{message} <> edges, fuel, True{}}

@unsafe
def relationLoop(+exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, +env: EvalEnv, choices: +List<Choice>, outputs: +List<U32>, predicate: U32, result: ChoiceResult) -> ChoiceResult:
  match choices result:
    case _ ChoiceResult{edges, fuel, True{}}: ChoiceResult{edges, fuel, True{}}
    case Nil{} ChoiceResult{edges, +fuel, False{}}:
      relationLeafBudget(U32.is_zero(fuel), exprs, cells, vars, env, outputs, predicate, ChoiceResult{edges, fuel, False{}})
    case Choice{slot, expr, membership} <> rest _:
      relationChoice(eval(exprs, env, expr), membership, slot, exprs, cells, vars, env, rest, outputs, predicate, result)

@unsafe
def relationChoice(value: Val, membership: Bool, slot: U32, exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, env: EvalEnv, rest: +List<Choice>, outputs: +List<U32>, predicate: U32, result: ChoiceResult) -> ChoiceResult:
  match value:
    case VE{message}: relationFailure(result, message)
    case _: relationChoiceKind(membership, value, slot, exprs, cells, vars, env, rest, outputs, predicate, result)

@unsafe
def relationChoiceKind(membership: Bool, value: Val, slot: U32, exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, env: EvalEnv, rest: +List<Choice>, outputs: +List<U32>, predicate: U32, result: ChoiceResult) -> ChoiceResult:
  match membership value:
    case False{} _: relationLoop(exprs, cells, vars, bindLocal(env, slot, value), rest, outputs, predicate, result)
    case True{} VSet{items, _}: relationValues(items, slot, exprs, cells, vars, env, rest, outputs, predicate, result)
    case _ _: relationFailure(result, "relation choice requires a finite set")

@unsafe
def relationValues(values: +List<Val>, +slot: U32, +exprs: +List<IExpr>, +cells: +List<Cell>, +vars: +List<Var>, +env: EvalEnv, +rest: +List<Choice>, +outputs: +List<U32>, +predicate: U32, result: ChoiceResult) -> ChoiceResult:
  match values result:
    case _ ChoiceResult{edges, fuel, True{}}: ChoiceResult{edges, fuel, True{}}
    case Nil{} _: result
    case value <> more _:
      relationValues(more, slot, exprs, cells, vars, env, rest, outputs, predicate,
        relationLoop(exprs, cells, vars, bindLocal(env, slot, value), rest, outputs, predicate, result))

@unsafe
def relationLeafBudget(exhausted: Bool, +exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, +env: EvalEnv, outputs: +List<U32>, predicate: U32, result: ChoiceResult) -> ChoiceResult:
  match exhausted:
    case True{}: relationFailure(result, "finite successor choices exceed supported allocation capacity")
    case False{}:
      relationPredicate(eval(exprs, env, predicate), exprs, cells, vars, env, outputs, result)

@unsafe
def relationPredicate(value: Val, exprs: +List<IExpr>, cells: +List<Cell>, vars: +List<Var>, env: EvalEnv, outputs: +List<U32>, result: ChoiceResult) -> ChoiceResult:
  match value result:
    case VB{False{}} ChoiceResult{edges, fuel, _}: ChoiceResult{edges, U32.sub(fuel, 1), False{}}
    case VB{True{}} _:
      relationEdge(edgeFromWhole(cells, vars, evalSetItems(exprs, env, outputs)), result)
    case VE{message} _: relationFailure(result, message)
    case _ _: relationFailure(result, "relation predicate is not boolean")

def relationEdge(edge: Core.Edge, result: ChoiceResult) -> ChoiceResult:
  match edge result:
    case Core.Rejected{message} _: relationFailure(result, message)
    case _ ChoiceResult{edges, fuel, _}: ChoiceResult{edge <> edges, U32.sub(fuel, 1), False{}}

# Whole-variable projection keeps foreign function key shapes visible instead
# of silently projecting them onto the packed TypeOK schema.
@unsafe
def edgeFromWhole(+cells: +List<Cell>, +vars: +List<Var>, +values: +List<Val>) -> Core.Edge:
  wholeError(firstErr(values), cells, vars, values)

@unsafe
def wholeError(error: Maybe<String>, +cells: +List<Cell>, +vars: +List<Var>, +values: +List<Val>) -> Core.Edge:
  match error:
    case Some{message}: Core.Rejected{message}
    case None{}: wholeProjected(projectWhole(vars, values, Projection{cells, Nil{}, True{}}), cells, vars, values)

@unsafe
def projectWhole(vars: +List<Var>, values: +List<Val>, projection: Projection) -> Projection:
  match vars values projection:
    case _ _ Projection{cells, reversed, False{}}: Projection{cells, reversed, False{}}
    case Nil{} Nil{} _: projection
    case Var{_, _, count, function} <> rest value <> more Projection{cells, reversed, _}:
      projectVariable(function, count, value, rest, more, cells, reversed)
    case _ _ Projection{cells, reversed, _}: Projection{cells, reversed, False{}}

@unsafe
def projectVariable(function: Bool, count: U32, value: Val, rest: +List<Var>, more: +List<Val>, cells: +List<Cell>, reversed: +List<Val>) -> Projection:
  match function:
    case False{}: projectScalar(value, rest, more, cells, reversed)
    case True{}: projectFunction(value, count, rest, more, cells, reversed)

@unsafe
def projectScalar(value: Val, rest: +List<Var>, more: +List<Val>, cells: +List<Cell>, reversed: +List<Val>) -> Projection:
  match cells:
    case _ <> remaining: projectWhole(rest, more, Projection{remaining, value <> reversed, True{}})
    case Nil{}: Projection{Nil{}, reversed, False{}}

@unsafe
def projectFunction(value: Val, +count: U32, rest: +List<Var>, more: +List<Val>, cells: +List<Cell>, reversed: +List<Val>) -> Projection:
  match value:
    case VFun{+tree, _}:
      projectFunctionSize(U32.is_eq(count, functionSize(tree)), count, tree, rest, more, cells, reversed)
    case _: Projection{cells, reversed, False{}}

@unsafe
def projectFunctionSize(equal: Bool, count: U32, tree: FunctionTree, rest: +List<Var>, more: +List<Val>, cells: +List<Cell>, reversed: +List<Val>) -> Projection:
  match equal:
    case False{}: Projection{cells, reversed, False{}}
    case True{}: projectWhole(rest, more, projectPacked(U32.to_nat(count), functionStack(tree, Nil{}), cells, reversed))

@unsafe
def projectPacked(count: Nat, stack: +List<FunctionTree>, cells: +List<Cell>, reversed: +List<Val>) -> Projection:
  match count stack cells:
    case 0n Nil{} _: Projection{cells, reversed, True{}}
    case 1n+rest FunctionNode{_, key, value, right, _, _} <> next Cell{_, Some{actual}, _, _} <> remaining:
      projectPackedKey(veq(key, actual), rest, functionStack(right, next), remaining, value, reversed)
    case _ _ _: Projection{cells, reversed, False{}}

@unsafe
def projectPackedKey(equal: Bool, count: Nat, stack: +List<FunctionTree>, cells: +List<Cell>, value: Val, reversed: +List<Val>) -> Projection:
  match equal:
    case True{}: projectPacked(count, stack, cells, value <> reversed)
    case False{}: Projection{cells, reversed, False{}}

@unsafe
def wholeProjected(projection: Projection, cells: +List<Cell>, +vars: +List<Var>, values: +List<Val>) -> Core.Edge:
  match projection:
    case Projection{Nil{}, reversed, True{}}: edgeOf(cells, vars, List.reverse(&2, Val, reversed))
    case _: Core.Outside{showWhole(vars, values)}

@unsafe
def wholeParts(vars: +List<Var>, values: +List<Val>) -> +List<String>:
  match vars values:
    case Var{name, _, _, _} <> rest value <> more:
      (Core.json_quote(name) ++ ":" ++ showVal(value)) <> wholeParts(rest, more)
    case _ _: Nil{}

def showWhole(vars: +List<Var>, values: +List<Val>) -> String:
  "{" ++ String.join(wholeParts(vars, values), ",") ++ "}"

# ENABLED is an existential question over the action's explicit choices, not
# over the TypeOK schema. Exhaustion/errors must never become \"disabled\".
type EnabledProgram is Data:
  EnabledProgram{guard: U32, choices: +List<Choice>, predicate: U32}

type EnabledResult is Data:
  EnabledResult{value: Val, fuel: U32}

type PrimeBinding is Data:
  PrimeScalar{slot: U32}
  PrimeFunction{slot: U32, keys: +List<Val>}

@unsafe def enabled(+exprs: +List<IExpr>, +env: EvalEnv, program: EnabledProgram) -> Val:
  match program:
    case EnabledProgram{guard, choices, predicate}:
      enabledGuard(eval(exprs, env, guard), exprs, env, choices, predicate)

@unsafe def enabledGuard(guard: Val, exprs: +List<IExpr>, env: EvalEnv, choices: +List<Choice>, predicate: U32) -> Val:
  match guard:
    case VB{False{}}: VB{False{}}
    case VB{True{}}: enabledValue(enabledLoop(exprs, env, choices, predicate, EnabledResult{VB{False{}}, Core.state_capacity()}))
    case VE{message}: VE{message}
    case _: VE{"enabled action guard is not Boolean"}

def enabledValue(result: EnabledResult) -> Val:
  match result:
    case EnabledResult{value, _}: value

def enabledError(result: EnabledResult, message: String) -> EnabledResult:
  match result:
    case EnabledResult{_, fuel}: EnabledResult{VE{message}, fuel}

@unsafe def enabledLoop(+exprs: +List<IExpr>, +env: EvalEnv, choices: +List<Choice>, predicate: U32, result: EnabledResult) -> EnabledResult:
  match choices result:
    case _ EnabledResult{VB{True{}}, fuel}: EnabledResult{VB{True{}}, fuel}
    case _ EnabledResult{VE{message}, fuel}: EnabledResult{VE{message}, fuel}
    case Nil{} EnabledResult{VB{False{}}, +fuel}:
      enabledBudget(U32.is_zero(fuel), exprs, env, predicate, fuel)
    case Choice{slot, expr, membership} <> rest _:
      enabledChoice(eval(exprs, env, expr), membership, slot, exprs, env, rest, predicate, result)
    case _ _: enabledError(result, "invalid enabled action search result")

@unsafe def enabledBudget(exhausted: Bool, exprs: +List<IExpr>, env: EvalEnv, predicate: U32, fuel: U32) -> EnabledResult:
  match exhausted:
    case True{}: EnabledResult{VE{"enabled action choices exceed supported allocation capacity"}, fuel}
    case False{}: enabledPredicate(eval(exprs, env, predicate), fuel)

def enabledPredicate(value: Val, fuel: U32) -> EnabledResult:
  match value:
    case VB{truth}: EnabledResult{VB{truth}, U32.sub(fuel, 1)}
    case VE{message}: EnabledResult{VE{message}, fuel}
    case _: EnabledResult{VE{"enabled action predicate is not Boolean"}, fuel}

@unsafe def enabledChoice(value: Val, membership: Bool, slot: U32, exprs: +List<IExpr>, env: EvalEnv, rest: +List<Choice>, predicate: U32, result: EnabledResult) -> EnabledResult:
  match value:
    case VE{message}: enabledError(result, message)
    case _: enabledChoiceKind(membership, value, slot, exprs, env, rest, predicate, result)

@unsafe def enabledChoiceKind(membership: Bool, value: Val, slot: U32, exprs: +List<IExpr>, env: EvalEnv, rest: +List<Choice>, predicate: U32, result: EnabledResult) -> EnabledResult:
  match membership value:
    case False{} _: enabledLoop(exprs, bindLocal(env, slot, value), rest, predicate, result)
    case True{} VSet{values, _}: enabledValues(values, slot, exprs, env, rest, predicate, result)
    case _ _: enabledError(result, "enabled action choice requires a finite set")

@unsafe def enabledValues(values: +List<Val>, +slot: U32, +exprs: +List<IExpr>, +env: EvalEnv, +rest: +List<Choice>, +predicate: U32, result: EnabledResult) -> EnabledResult:
  match values result:
    case _ EnabledResult{VB{True{}}, fuel}: EnabledResult{VB{True{}}, fuel}
    case _ EnabledResult{VE{message}, fuel}: EnabledResult{VE{message}, fuel}
    case Nil{} _: result
    case value <> more _:
      enabledValues(more, slot, exprs, env, rest, predicate, enabledLoop(exprs, bindLocal(env, slot, value), rest, predicate, result))

# Bind whole next-variable values in one pass over decoded cells. Packed
# functions are reconstructed in linear time, never by repeated slot lookup.
@unsafe def primeEnv(bindings: +List<PrimeBinding>, values: +List<Val>, env: EvalEnv) -> Result<&2, &2, String, EvalEnv>:
  match bindings values:
    case Nil{} Nil{}: Done{env}
    case PrimeScalar{slot} <> rest value <> more: primeEnv(rest, more, bindLocal(env, slot, value))
    case PrimeFunction{slot, +keys} <> rest _: primeFunction(keys, keys, values, slot, rest, env, Nil{})
    case _ _: Fail{"primed variable bindings do not match the decoded next valuation"}

@unsafe def primeFunction(+keys: +List<Val>, remaining: +List<Val>, values: +List<Val>, slot: U32, rest: +List<PrimeBinding>, env: EvalEnv, reversed: +List<Val>) -> Result<&2, &2, String, EvalEnv>:
  match remaining values:
    case Nil{} _: primeFunctionValue(functionSorted(keys, List.reverse(&2, Val, reversed)), values, slot, rest, env)
    case _ <> more value <> tail: primeFunction(keys, more, tail, slot, rest, env, value <> reversed)
    case _ Nil{}: Fail{"packed primed function exceeds the decoded next valuation"}

@unsafe def primeFunctionValue(value: Val, values: +List<Val>, slot: U32, rest: +List<PrimeBinding>, env: EvalEnv) -> Result<&2, &2, String, EvalEnv>:
  match value:
    case VE{message}: Fail{message}
    case _: primeEnv(rest, values, bindLocal(env, slot, value))
