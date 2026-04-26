# Where does zou3519's redispatch-chain prototype lose ~5μs vs the current PR fast path?

**TL;DR.** ~96% of the gap is three structural costs of the redispatch-chain shape, not predicate recompute:

| Step (no-grad path) | median ns | IQR | % of gap |
|---|---:|---:|---:|
| `op.redispatch` chain pop (TLS read + `chain.pop()` + indirect Python call + `_enable_fast_dispatch` ctx enter/exit) | **2,108** | 137 | **40%** |
| `backend_dispatch` Python frame (extra call + redundant `args[0].device.type` and `raw_fns.get` lookups) | **1,810** | 584 | **34%** |
| `_AutoDispatchBelowAutograd` ctx enter/exit (+ `keyset & after_autograd`) | **1,226** | 813 | **23%** |
| `Metadata(keyset, {})` dataclass alloc | 301 | 423 | within noise |
| `forward_no_grad` frame | 224 | 347 | within noise |
| `autograd_impl` frame (TLS-shortcut form) | 200 | 295 | within noise |
| Predicate recompute (`is_grad_enabled() and _any_requires_grad(*args)`) | 145 | 227 | within noise |

Sum of medians from `v1` (current-PR-shape) to `v8` (zou3519-prototype-shape) = **6,015 ns**, vs the directly measured Apr 23 gap of **5,284 ns**. The 700 ns excess is variant-scaffolding overhead (closure indirection through `body(args, fn, device, any_grad)` that the unmodified paths don't have).

So **zou3519's TLS-shortcut idea saves essentially nothing on the no-grad path**. `is_grad_enabled()` is a cheap C++ flag read; `_any_requires_grad` short-circuits at the first non-grad tensor. The cost lives downstream.

## Method

Built nine no-grad variants. Each variant's `fast_call` shares the same prelude (the `_C._custom_op_fast_path_check` C++ guard, kwargs/disabled/fn lookup); they differ only in the no-grad body, building **up** from current-PR shape to zou3519-prototype shape one ingredient at a time:

| Variant | Body (post-prelude) |
|---|---|
| `v0_baseline` | `fn(*args)` (kernel only) |
| `v1_alias` | + `_c_check_aliasing_constraint` ≈ current PR no-grad fast path |
| `v2_backend_dispatch` | + extra Python frame doing redundant device.type/fn lookup |
| `v3_redispatch` | + `with _enable_fast_dispatch(...): op.redispatch(...)` (chain pop) |
| `v4_autodisp` | + `_AutoDispatchBelowAutograd` ctx + `keyset & after_autograd` |
| `v5_metadata` | + `Metadata(keyset, {})` dataclass alloc |
| `v6_forward_no_grad` | + Python frame for `forward_no_grad` |
| `v7_autograd_impl_sc` | + `autograd_impl` frame, TLS-shortcut (no predicate recompute) |
| `v8_autograd_impl_full` | + recompute `is_grad_enabled() and _any_requires_grad(*args)` (= zou3519's prototype) |

Variant deltas ≈ cost of each ingredient.

Built on `reduce-custom_op-overhead` so `_C._custom_op_fast_path_check` is available. `OpOverload.redispatch` is monkey-patched in the harness to chain-pop when TLS is set (mirrors zou3519's prototype's `_enable_fast_dispatch`). Each variant is installed as `nonmut._fast_call` and benchmarked via `nonmut(x)` with `x = torch.randn(4)`. Single GB200 ARM core (`taskset -c 8`), performance governor, fixed 3.366 GHz.

Per call: `min(30k iters, 11 runs)`, in-process, fixed v0→v8 order. Five back-to-back invocations to control for absolute drift; numbers above are medians (runs 4 minutes total).

## Validation against Apr 23 baseline

| | Apr 23 directly | this profile |
|---|---:|---:|
| current PR no-grad | 3,451 | 3,873 (`v1_alias` median) |
| zou3519 prototype no-grad | 8,735 | 8,422 (`v8_autograd_impl_full` median) |
| **gap** | **5,284** | **4,549 across the 5-run medians, 6,015 by summed step medians** |

`v1` is ~400 ns slower than the real PR fast path due to closure indirection in `body(...)`; `v8` is ~300 ns faster than the real prototype for the same reason. They roughly cancel and the per-step deltas track the underlying cost.

## What this means for the design choice

To recover ~5μs you have to attack the redispatch-chain shape itself: kill the `op.redispatch` hop (~40%), kill the `backend_dispatch` frame (~34%), kill `_AutoDispatchBelowAutograd` (~23%) — i.e. inline the kernel call in `fast_call`. That's exactly what the current PR does.

A TLS bool to skip predicate recompute saves ~150 ns and is dwarfed by everything else.

## Files

- `profile_redispatch_chain.py` — variant builder + benchmark harness. Prefer running in-process (`python profile_redispatch_chain.py`); subprocess-isolated mode (`--mode v3_redispatch`) is also wired up but the in-process fixed-order runs are more stable in practice.
- `aggregate.py` — pulls per-step deltas out of a multi-run log and prints median/min/max/IQR.
- `result_5runs_in_order.txt` — raw output of the five back-to-back runs the table above is computed from.
- `result_run1.txt` … `result_run4.txt` — earlier exploratory runs (with shuffling, isolated subprocesses); kept for reference but not the primary signal.
