import Base

# IDs are exact mixed-radix state identities, not fingerprints.
type Edge is Data:
  To{state: U32}
  Outside{json: String}
  Rejected{message: String}

# Safety-only searches allocate no graph. Temporal searches retain exact,
# deduplicated non-stuttering edges under the same single affine owner.
type GraphMode is Type:
  NoGraph{}
  RecordedGraph{rows: Array<+List<U32>>, marks: Array<U32>, edges: U32, count: U32}

type GraphSnapshot is Type:
  GraphSnapshot{rows: Array<+List<U32>>, parents: Array<U32>, queue: Array<U32>, states: U32, count: U32}

type RowBuild is Type:
  RowBuild{rows: Array<+List<U32>>, marks: Array<U32>, edges: U32, count: U32, source: U32, reversed: +List<U32>}

type InvariantCheck is Data:
  InvariantHolds{}
  InvariantViolation{name: String}
  InvariantError{message: String}

type Store is Type:
  Store{parents: Array<U32>, queue: Array<U32>, head: U32, tail: U32, check_deadlock: Bool, graph: GraphMode}

type Report is Type:
  SafeReport{states: U32}
  ViolationReport{states: U32, invariant: String, trace: List<U32>, outside: Maybe<String>}
  DeadlockReport{states: U32, trace: List<U32>}
  GraphReport{snapshot: GraphSnapshot}
  TemporalViolationReport{states: U32, property: String, prefix: List<U32>, cycle: List<U32>}
  ErrorReport{message: String}

type Trail is Type:
  Trail{parents: Array<U32>, state: U32, acc: List<U32>}

type TraceStop is Type:
  TraceFound{trace: List<U32>}
  TraceFault{message: String}

def initial_graph(enabled: Bool, count: U32, +depth: Nat) -> GraphMode:
  match enabled:
    case False{}: NoGraph{}
    case True{}:
      RecordedGraph{Array.new(+List<U32>, depth, Nil{}), Array.new(U32, depth, 4294967295), 0, count}

def finish_graph(graph: GraphMode, parents: Array<U32>, queue: Array<U32>, states: U32) -> Report:
  match graph:
    case NoGraph{}: SafeReport{states}
    case RecordedGraph{rows, _, _, count}: GraphReport{GraphSnapshot{rows, parents, queue, states, count}}

def report_store(store: Store) -> Report:
  match store:
    case Store{parents, queue, _, tail, _, graph}: finish_graph(graph, parents, queue, tail)

@unsafe def record_graph(graph: GraphMode, +source: U32, edges: +List<Edge>) -> Result<&1, &1, Report, GraphMode>:
  match graph:
    case NoGraph{}: Done{NoGraph{}}
    case RecordedGraph{rows, marks, count_edges, count}:
      graph_row(edges, RowBuild{rows, Array.set(U32, marks, source, source), count_edges, count, source, Nil{}})

@unsafe def graph_row(edges: +List<Edge>, row: RowBuild) -> Result<&1, &1, Report, GraphMode>:
  match edges row:
    case Nil{} RowBuild{rows, marks, count_edges, count, source, reversed}:
      Done{RecordedGraph{Array.set(+List<U32>, rows, source, List.reverse(&2, U32, reversed)), marks, count_edges, count}}
    case To{+target} <> rest RowBuild{rows, marks, count_edges, +count, source, reversed}:
      graph_target_valid(U32.is_lt(target, count), target, rest, RowBuild{rows, marks, count_edges, count, source, reversed})
    case _ <> rest _: graph_row(rest, row)

@unsafe def graph_target_valid(valid: Bool, +target: U32, rest: +List<Edge>, row: RowBuild) -> Result<&1, &1, Report, GraphMode>:
  match valid row:
    case False{} _: Fail{ErrorReport{"successor ID is outside the exact state universe"}}
    case True{} RowBuild{rows, marks, count_edges, count, source, reversed}:
      graph_mark_read(Array.get(U32, marks, target), target, rest, rows, count_edges, count, source, reversed)

@unsafe def graph_mark_read(result: Array<U32> & U32, target: U32, rest: +List<Edge>, rows: Array<+List<U32>>, count_edges: U32, count: U32, +source: U32, reversed: +List<U32>) -> Result<&1, &1, Report, GraphMode>:
  match result:
    case (marks, previous):
      graph_mark_ready(U32.is_eq(previous, source), target, rest, RowBuild{rows, marks, count_edges, count, source, reversed})

@unsafe def graph_mark_ready(seen: Bool, target: U32, rest: +List<Edge>, row: RowBuild) -> Result<&1, &1, Report, GraphMode>:
  match seen row:
    case True{} _: graph_row(rest, row)
    case False{} RowBuild{rows, marks, +count_edges, count, source, reversed}:
      graph_edge_capacity(U32.is_ge(count_edges, state_capacity()), target, rest, RowBuild{rows, marks, count_edges, count, source, reversed})

@unsafe def graph_edge_capacity(full: Bool, +target: U32, rest: +List<Edge>, row: RowBuild) -> Result<&1, &1, Report, GraphMode>:
  match full row:
    case True{} _: Fail{ErrorReport{"temporal transition graph exceeds supported allocation capacity"}}
    case False{} RowBuild{rows, marks, count_edges, count, +source, reversed}:
      graph_row(rest, RowBuild{rows, Array.set(U32, marks, target, source), U32.inc(count_edges), count, source, target <> reversed})
def hex_digit(n: U32, small: Bool) -> Char:
  match small:
    case True{}:
      Chr{(n + 48 : U32)}
    case False{}:
      Chr{(n + 87 : U32)}

def json_control(+n: U32, control: Bool) -> String:
  match control:
    case False{}:
      Char.show(Chr{n})
    case True{}:
      +hi = (n / 16 : U32)
      +lo = (n % 16 : U32)
      "\\u00" ++ Char.show(hex_digit(hi, (hi < 10 : U32))) ++ Char.show(hex_digit(lo, (lo < 10 : U32)))

def json_char(c: Char) -> String:
  match c:
    case Chr{34}:
      "\\\""
    case Chr{92}:
      "\\\\"
    case Chr{+n}:
      json_control(n, (n < 32 : U32))

def json_body(s: String) -> String:
  match s:
    case SNil{}:
      "\""
    case SCon{h, tail}:
      json_char(h) ++ json_body(tail)

def json_quote(s: String) -> String:
  "\"" ++ json_body(s)

def trace_link(root: Bool, valid: Bool, parents: Array<U32>, previous: U32, trace: List<U32>) -> Result<&1, &1, TraceStop, Trail>:
  match root valid:
    case True{} _:
      Fail{TraceFound{trace}}
    case False{} True{}:
      Done{Trail{parents, previous, trace}}
    case False{} False{}:
      Fail{TraceFault{"invalid predecessor in exact state store"}}

def trace_read(result: Array<U32> & U32, +state: U32, count: U32, acc: List<U32>) -> Result<&1, &1, TraceStop, Trail>:
  match result:
    case (parents, +previous):
      trace_link(U32.is_eq(previous, state), (previous < count : U32), parents, previous, state <> acc)

def trace_step(cursor: Trail, count: U32) -> Result<&1, &1, TraceStop, Trail>:
  match cursor:
    case Trail{parents, +state, acc}:
      trace_read(Array.get(U32, parents, state), state, count, acc)

def trace_loop(fuel: Nat, cursor: Trail, +count: U32) -> Result<&1, &1, TraceStop, Trail>:
  match fuel:
    case 0n:
      Fail{TraceFault{"predecessor chain exceeds the finite state universe"}}
    case 1n+rest:
      do Result<&1, &1, TraceStop, Trail>:
        after : Trail <- trace_step(cursor, count)
        trace_loop(rest, after, count)

def finish_trace(result: Result<&1, &1, TraceStop, Trail>, states: U32, name: String, outside: Maybe<String>) -> Report:
  match result:
    case Fail{TraceFound{trace}}:
      ViolationReport{states, name, trace, outside}
    case Fail{TraceFault{message}}:
      ErrorReport{message}
    case Done{_}:
      ErrorReport{"unfinished predecessor traversal"}

def violation(parents: Array<U32>, parent: Maybe<&2, U32>, +count: U32, states: U32, name: String, outside: Maybe<String>) -> Report:
  match parent:
    case None{}:
      ViolationReport{states, name, Nil{}, outside}
    case Some{state}:
      finish_trace(trace_loop(U32.to_nat(count), Trail{parents, state, Nil{}}, count), states, name, outside)

def deadlock_trace(result: Result<&1, &1, TraceStop, Trail>, states: U32) -> Report:
  match result:
    case Fail{TraceFound{trace}}: DeadlockReport{states, trace}
    case Fail{TraceFault{message}}: ErrorReport{message}
    case Done{_}: ErrorReport{"unfinished deadlock predecessor traversal"}

def deadlock_report(parents: Array<U32>, state: U32, +count: U32, states: U32) -> Report:
  deadlock_trace(trace_loop(U32.to_nat(count), Trail{parents, state, Nil{}}, count), states)

def parent_id(state: U32, parent: Maybe<&2, U32>) -> U32:
  match parent:
    case None{}:
      state
    case Some{id}:
      id

def after_invariant(result: InvariantCheck, store: Store, state: U32, count: U32) -> Result<&1, &1, Report, Store>:
  match result:
    case InvariantHolds{}:
      Done{store}
    case InvariantViolation{name}:
      Store{parents, _, _, tail, _, _} = store
      Fail{violation(parents, Some{state}, count, tail, name, None{})}
    case InvariantError{message}:
      Fail{ErrorReport{message}}

def fresh_state(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, available: Bool, +env: Env, +state: U32, parent: Maybe<&2, U32>, count: U32, store: Store) -> Result<&1, &1, Report, Store>:
  match available:
    case False{}:
      Fail{ErrorReport{"exact state queue capacity exceeded"}}
    case True{}:
      Store{parents, queue, head, +tail, check_deadlock, graph} = store
      parents = Array.set(U32, parents, state, parent_id(state, parent))
      queue = Array.set(U32, queue, tail, state)
      after_invariant(invariant(env, state), Store{parents, queue, head, (tail + 1 : U32), check_deadlock, graph}, state, count)

def visited_state(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, seen: Bool, env: Env, state: U32, parent: Maybe<&2, U32>, +count: U32, store: Store) -> Result<&1, &1, Report, Store>:
  match seen:
    case True{}:
      Done{store}
    case False{}:
      Store{parents, queue, head, +tail, check_deadlock, graph} = store
      fresh_state(~Env, ~invariant, (tail < count : U32), env, state, parent, count, Store{parents, queue, head, tail, check_deadlock, graph})

def visited_read(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, result: Array<U32> & U32, queue: Array<U32>, head: U32, tail: U32, env: Env, state: U32, parent: Maybe<&2, U32>, count: U32, check_deadlock: Bool, graph: GraphMode) -> Result<&1, &1, Report, Store>:
  match result:
    case (parents, previous):
      visited_state(~Env, ~invariant, U32.is_ne(previous, 4294967295), env, state, parent, count, Store{parents, queue, head, tail, check_deadlock, graph})

def valid_state(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, valid: Bool, env: Env, +state: U32, parent: Maybe<&2, U32>, count: U32, store: Store) -> Result<&1, &1, Report, Store>:
  match valid:
    case False{}:
      Fail{ErrorReport{"successor ID is outside the exact state universe"}}
    case True{}:
      Store{parents, queue, head, tail, check_deadlock, graph} = store
      visited_read(~Env, ~invariant, Array.get(U32, parents, state), queue, head, tail, env, state, parent, count, check_deadlock, graph)

def discover(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, edge: Edge, env: Env, parent: Maybe<&2, U32>, +count: U32, store: Store) -> Result<&1, &1, Report, Store>:
  match edge:
    case To{+state}:
      valid_state(~Env, ~invariant, (state < count : U32), env, state, parent, count, store)
    case Outside{json}:
      Store{parents, _, _, tail, _, _} = store
      Fail{violation(parents, parent, count, tail, "TypeOK", Some{json})}
    case Rejected{message}:
      Fail{ErrorReport{message}}

# Keep affine store loops first-order. Generic Result.bind continuations inherit
# the native scheduler's global closure fork classification.
def discover_many(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, edges: +List<Edge>, +env: Env, +parent: Maybe<&2, U32>, +count: U32, result: Result<&1, &1, Report, Store>) -> Result<&1, &1, Report, Store>:
  match edges result:
    case _ Fail{report}:
      Fail{report}
    case Nil{} Done{store}:
      Done{store}
    case Con{edge, rest} Done{store}:
      discover_many(~Env, ~invariant, rest, env, parent, count, discover(~Env, ~invariant, edge, env, parent, count, store))

def recorded_edges(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, result: Result<&1, &1, Report, GraphMode>, edges: +List<Edge>, env: Env, +state: U32, count: U32, parents: Array<U32>, queue: Array<U32>, head: U32, tail: U32, check_deadlock: Bool) -> Result<&1, &1, Report, Store>:
  match result:
    case Fail{report}: Fail{report}
    case Done{graph}:
      discover_many(~Env, ~invariant, edges, env, Some{state}, count, Done{Store{parents, queue, head, tail, check_deadlock, graph}})


def blocked_state(check_deadlock: Bool, parents: Array<U32>, queue: Array<U32>, head: U32, tail: U32, state: U32, count: U32, graph: GraphMode) -> Result<&1, &1, Report, Store>:
  match check_deadlock:
    case True{}: Fail{deadlock_report(parents, state, count, tail)}
    case False{}: Done{Store{parents, queue, head, tail, False{}, graph}}


@unsafe def explore_graph(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, graph: GraphMode, +edges: +List<Edge>, env: Env, +state: U32, count: U32, parents: Array<U32>, queue: Array<U32>, head: U32, tail: U32, check_deadlock: Bool) -> Result<&1, &1, Report, Store>:
  match graph:
    case NoGraph{}:
      discover_many(~Env, ~invariant, edges, env, Some{state}, count, Done{Store{parents, queue, head, tail, check_deadlock, NoGraph{}}})
    case _:
      recorded_edges(~Env, ~invariant, record_graph(graph, state, edges), edges, env, state, count, parents, queue, head, tail, check_deadlock)

def explore_edges(~Env: Data, ~invariant: Env -> U32 -> InvariantCheck, +edges: +List<Edge>, env: Env, +state: U32, count: U32, store: Store) -> Result<&1, &1, Report, Store>:
  match edges:
    case Nil{}:
      Store{parents, queue, head, tail, check_deadlock, graph} = store
      blocked_state(check_deadlock, parents, queue, head, tail, state, count, graph)
    case _:
      Store{parents, queue, head, tail, check_deadlock, graph} = store
      explore_graph(~Env, ~invariant, graph, edges, env, state, count, parents, queue, head, tail, check_deadlock)
def explore_read(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, result: Array<U32> & U32, parents: Array<U32>, head: U32, tail: U32, +env: Env, count: U32, check_deadlock: Bool, graph: GraphMode) -> Result<&1, &1, Report, Store>:
  match result:
    case (queue, +state):
      explore_edges(~Env, ~invariant, next(env, state), env, state, count, Store{parents, queue, (head + 1 : U32), tail, check_deadlock, graph})

def explore_ready(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, empty: Bool, +env: Env, count: U32, store: Store) -> Result<&1, &1, Report, Store>:
  match empty:
    case True{}:
      Fail{report_store(store)}
    case False{}:
      Store{parents, queue, +head, tail, check_deadlock, graph} = store
      explore_read(~Env, ~next, ~invariant, Array.get(U32, queue, head), parents, head, tail, env, count, check_deadlock, graph)

def explore_step(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, env: Env, count: U32, store: Store) -> Result<&1, &1, Report, Store>:
  match store:
    case Store{parents, queue, +head, +tail, check_deadlock, graph}:
      explore_ready(~Env, ~next, ~invariant, U32.is_eq(head, tail), env, count, Store{parents, queue, head, tail, check_deadlock, graph})

def finish_empty(empty: Bool, store: Store) -> Result<&1, &1, Report, Store>:
  match empty:
    case True{}: Fail{report_store(store)}
    case False{}: Fail{ErrorReport{"finite state fuel exhausted with a nonempty queue"}}

def finish_exploration(store: Store) -> Result<&1, &1, Report, Store>:
  match store:
    case Store{parents, queue, +head, +tail, check_deadlock, graph}:
      finish_empty(U32.is_eq(head, tail), Store{parents, queue, head, tail, check_deadlock, graph})

def explore(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, fuel: Nat, +env: Env, +count: U32, result: Result<&1, &1, Report, Store>) -> Result<&1, &1, Report, Store>:
  match fuel result:
    case _ Fail{report}:
      Fail{report}
    case 0n Done{store}:
      finish_exploration(store)
    case 1n+rest Done{store}:
      explore(~Env, ~next, ~invariant, rest, env, count, explore_step(~Env, ~next, ~invariant, env, count, store))

def unwrap_report(result: Result<&1, &1, Report, Store>) -> Report:
  match result:
    case Fail{report}:
      report
    case Done{_}:
      ErrorReport{"unfinished reachable-state exploration"}

def seeded_ready(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, empty: Bool, env: Env, +count: U32, store: Store) -> Report:
  match empty:
    case True{}:
      ErrorReport{"Init has no states"}
    case False{}:
      unwrap_report(explore(~Env, ~next, ~invariant, U32.to_nat(count), env, count, Done{store}))

def seeded(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, result: Result<&1, &1, Report, Store>, env: Env, count: U32) -> Report:
  match result:
    case Fail{report}:
      report
    case Done{Store{parents, queue, head, +tail, check_deadlock, graph}}:
      seeded_ready(~Env, ~next, ~invariant, U32.is_zero(tail), env, count, Store{parents, queue, head, tail, check_deadlock, graph})

def state_capacity() -> U32:
  1048576

# ceil(log2(count)), bounded structurally by the width of a state ID.
def depth_bits(fuel: Nat, value: U32, depth: Nat) -> Nat:
  match fuel:
    case 0n: depth
    case 1n+rest:
      match value:
        case 0: depth
        case bits: depth_bits(rest, U32.shr(bits), 1n+depth)

def storage_depth(count: U32) -> Nat:
  depth_bits(32n, U32.sub(count, 1), 0n)

def search(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, +env: Env, initial: +List<Edge>, +count: U32, +depth: Nat, check_deadlock: Bool, record: Bool) -> Report:
  parents = Array.new(U32, depth, 4294967295)
  queue = Array.new(U32, depth, 0)
  seeded(~Env, ~next, ~invariant, discover_many(~Env, ~invariant, initial, env, None{}, count, Done{Store{parents, queue, 0, 0, check_deadlock, initial_graph(record, count, depth)}}), env, count)

def outside_json(outside: Maybe<String>, comma: Bool) -> String:
  match outside comma:
    case None{} _:
      ""
    case Some{json} False{}:
      json
    case Some{json} True{}:
      "," ++ json

def trace_json(~Env: Data, ~show: Env -> U32 -> String, trace: List<U32>, +env: Env, outside: Maybe<String>) -> String:
  match trace:
    case Nil{}:
      outside_json(outside, False{})
    case Con{state, Nil{}}:
      show(env, state) ++ outside_json(outside, True{})
    case Con{state, rest}:
      show(env, state) ++ "," ++ trace_json(~Env, ~show, rest, env, outside)

def render(~Env: Data, ~show: Env -> U32 -> String, report: Report, +env: Env) -> String & U32:
  match report:
    case SafeReport{states}:
      ("{\"status\":\"safe\",\"states\":" ++ U32.show(states) ++ "}", 0)
    case ViolationReport{states, name, trace, outside}:
      ("{\"status\":\"violation\",\"states\":" ++ U32.show(states) ++ ",\"invariant\":" ++ json_quote(name) ++ ",\"trace\":[" ++ trace_json(~Env, ~show, trace, env, outside) ++ "]}", 1)
    case DeadlockReport{states, trace}:
      ("{\"status\":\"deadlock\",\"states\":" ++ U32.show(states) ++ ",\"trace\":[" ++ trace_json(~Env, ~show, trace, env, None{}) ++ "]}", 1)
    case TemporalViolationReport{states, property, prefix, cycle}:
      ("{\"status\":\"temporal-violation\",\"states\":" ++ U32.show(states) ++ ",\"property\":" ++ json_quote(property) ++ ",\"prefix\":[" ++ trace_json(~Env, ~show, prefix, env, None{}) ++ "],\"cycle\":[" ++ trace_json(~Env, ~show, cycle, env, None{}) ++ "]}", 1)
    case GraphReport{_}:
      ("{\"status\":\"error\",\"message\":\"internal transition graph reached the report renderer\"}", 2)
    case ErrorReport{message}:
      ("{\"status\":\"error\",\"message\":" ++ json_quote(message) ++ "}", 2)

def emit(result: String & U32) -> IO(Unit):
  match result:
    case (json, code):
      do IO<Unit>:
        IO.print(json)
        IO.exit(code)

def run_valid(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, ~show: Env -> U32 -> String, valid: Bool, +env: Env, initial: +List<Edge>, +count: U32, check_deadlock: Bool) -> IO(Unit):
  match valid:
    case False{}:
      emit(render(~Env, ~show, ErrorReport{"state universe exceeds supported allocation capacity"}, env))
    case True{}:
      emit(render(~Env, ~show, search(~Env, ~next, ~invariant, env, initial, count, storage_depth(count), check_deadlock, False{}), env))

def run(~Env: Data, ~next: Env -> U32 -> +List<Edge>, ~invariant: Env -> U32 -> InvariantCheck, ~show: Env -> U32 -> String, +env: Env, initial: +List<Edge>, +count: U32, check_deadlock: Bool) -> IO(Unit):
  run_valid(~Env, ~next, ~invariant, ~show, (count > 0 : U32) && (count <= state_capacity() : U32), env, initial, count, check_deadlock)
