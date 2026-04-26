# Where does zou3519's redispatch-chain prototype lose ~5μs?

Profile harness for the no-grad path of `torch.library.custom_op`'s fast path. Compares the current PR (`reduce-custom_op-overhead`, inlined kernel call) against zou3519's prototype (`zou3519`'s "redispatch chain in Python", commit `2bf17e99a1d`).

## Method — *not* "short-circuit at a profile point"

I did **not** instrument the actual prototype's `autograd_impl` and short-circuit it at increasing depths. That approach has two problems: (1) it measures *everything before* the short-circuit, including any overhead from instrumentation itself, and (2) it can't easily reach below `_AutoDispatchBelowAutograd` or below `op.redispatch` because those are C++/dispatch boundaries.

Instead, I built **nine variants of the no-grad path bottom-up**. For each variant the end-to-end route is owned by one variant body — disjoint paths, only one runs per call. The deeper variants don't inline everything into one function; they call through the same helper frames the prototype uses (`autograd_impl` → `forward_no_grad` → `op.redispatch` → `backend_dispatch`), so Python frame-creation costs are paid where the prototype pays them. They all share the same prelude (the `_C._custom_op_fast_path_check` C++ guard, kwargs/disabled/fn lookup), so prelude cost cancels in every delta.

Each variant adds one ingredient on top of its predecessor:

```
v0_baseline             :   fn(*args)                                          # kernel only
v1_alias                : + _c_check_aliasing_constraint                       # current PR no-grad fast path
v2_backend_dispatch     : + Python frame doing redundant device.type/fn lookup
v3_redispatch           : + _enable_fast_dispatch ctx + op.redispatch chain pop
v4_autodisp             : + _AutoDispatchBelowAutograd + (keyset & after_autograd)
v5_metadata             : + Metadata(keyset, {}) dataclass alloc
v6_forward_no_grad      : + Python frame for forward_no_grad
v7_autograd_impl_sc     : + autograd_impl frame, TLS-shortcut (no predicate recompute)
v8_autograd_impl_full   : + recompute is_grad_enabled() and _any_requires_grad(*args)  -- == zou3519's prototype
```

Concretely, each variant is a closure produced by `make_variant(opdef, mode)` (see `profile_redispatch_chain.py`). The closure has the same signature as a normal `fast_call`:

```python
def fast_call(*args, **kwargs):
    # SHARED prelude (identical across all variants):
    if torch.compiler.is_compiling(): return _FALLBACK
    if not args or kwargs: return _FALLBACK
    check = _C._custom_op_fast_path_check(args)
    if check is None: return _FALLBACK
    device_type, any_requires_grad = check
    if device_type == "meta" or device_type in disabled: return _FALLBACK
    fn = raw_fns.get(device_type) or raw_fns.get(None)
    if fn is None: return _FALLBACK
    if torch.is_grad_enabled() and any_requires_grad: return _FALLBACK
    # VARIANT-SPECIFIC body — this is the only thing that changes:
    return body(args, fn, device_type, any_requires_grad)
```

The variant body is installed via `nonmut._fast_call = fast_call`, which is what `CustomOpDef.__call__` invokes for the user-API call `nonmut(x)`. The benchmark loop calls `nonmut(x)` repeatedly and times it.

Because every variant goes through the exact same prelude and only the body differs, `v_n - v_{n-1}` measures the cost of the one ingredient added in variant `n`.

### Why bottom-up vs top-down?

Top-down (short-circuit zou3519's full path at progressively earlier points) gives you the cumulative cost at each profile point. Bottom-up (this approach) gives you the *marginal* cost of each ingredient. The marginal cost is what you actually want when deciding which ingredients to keep or remove.

Bottom-up also avoids a subtle bias: a top-down short-circuit added inside `autograd_impl` only measures what was *already going to run before the short-circuit*. It can't measure things downstream of `op.redispatch` (the chain pop, the backend frame, the alias check) without separately instrumenting those — which adds overhead that pollutes the measurement.

### Faithfulness to zou3519's prototype

To make `op.redispatch` chain-pop in Python instead of going to C++, the harness monkey-patches `OpOverload.redispatch` with the same logic as the prototype's commit:

```python
def _patched_redispatch(self, keyset, *args, **kwargs):
    entry = getattr(_fast_dispatch_tls, "entry", None)
    if entry is not None:
        target_op, chain = entry
        if target_op is self and chain:
            fn = chain.pop()
            return fn(keyset, *args, **kwargs)
    return _orig_redispatch(self, keyset, *args, **kwargs)
```

Validation: `v1_alias` ≈ current PR no-grad (3,873 vs 3,451 measured Apr 23, the ~400 ns extra is closure indirection from `body(...)`); `v8_autograd_impl_full` ≈ zou3519's prototype (8,422 vs 8,735 measured Apr 23, ~300 ns less for the same reason). They roughly cancel and the per-step deltas track the underlying cost.

## Results

See [`RESULTS.md`](RESULTS.md). Three structural costs of the redispatch-chain shape account for ~96% of the gap:

| Step | median ns | % of gap |
|---|---:|---:|
| `op.redispatch` chain pop | 2,108 | 40% |
| `backend_dispatch` Python frame | 1,810 | 34% |
| `_AutoDispatchBelowAutograd` ctx | 1,226 | 23% |
| Predicate recompute | 145 | within noise |

## Reproduce

Need PyTorch built on `reduce-custom_op-overhead` (so `_C._custom_op_fast_path_check` is available).

```bash
# Pinned to a single core (any single core works; my measurements used core 8 of a GB200 ARM host).
taskset -c 8 python profile_redispatch_chain.py --repeats 11 --n 30000

# Reproduce the 5-back-to-back-runs that the table in RESULTS.md is computed from:
for i in 1 2 3 4 5; do
  echo "=== run $i ==="
  taskset -c 8 python profile_redispatch_chain.py --repeats 11 --n 30000 \
    | grep -E "(real|v[0-9]_)"
done > result_5runs.txt

# Aggregate per-step deltas (median / min / max / IQR across runs):
python aggregate.py result_5runs.txt
```

Notes on stability:
- In-process, fixed v0→v8 order is the most stable measurement mode I found. Subprocess-isolated mode (`--mode <name>`) and shuffled order (`--shuffle-seed`) are wired up too but were less stable in practice.
- Absolute numbers drift run-to-run by ~1–2 μs (CPU/system state); the *deltas* (which are what we care about) are stable to ~200 ns for the structural costs and within noise for the small ones.
- IQR for the dominant `v2→v3` delta is just 137 ns over 5 runs; it's the most reliable single number in the report.

## Files

- `profile_redispatch_chain.py` — variant builder + benchmark harness.
- `aggregate.py` — pulls per-step deltas out of a multi-run log and prints median/min/max/IQR.
- `run_isolated.sh` — alt harness running each variant in its own python subprocess. Less stable; kept for reference.
- `result_5runs_in_order.txt`, `result_run*.txt` — raw outputs. Apr 26 GB200 ARM core, performance governor, 3.366 GHz.
- `RESULTS.md` — final table + validation against the Apr 23 directly-measured baseline.
- `SLACK_REPLY.md` — drafted Slack reply.
