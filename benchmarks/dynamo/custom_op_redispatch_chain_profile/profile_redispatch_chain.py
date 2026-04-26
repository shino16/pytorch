"""Pin down where Richard's redispatch-chain prototype loses ~5μs vs current PR.

Builds the no-grad path *up* from current-PR shape to full-richard shape, one
piece at a time. Each delta = cost of that piece.

Variants (no-grad path body, after the shared C++ guard + device/fn lookup):

  v0_baseline           : just  fn(*args)                              # kernel only
  v1_alias              : fn(*args) + _c_check_aliasing_constraint    # current PR fast_call body
  v2_backend_dispatch   : + extra Python frame doing redundant device.type/fn lookup
  v3_redispatch         : + op.redispatch -> _enable_fast_dispatch chain pop
  v4_autodisp           : + _AutoDispatchBelowAutograd ctx
  v5_metadata           : + Metadata dataclass alloc
  v6_forward_no_grad    : + forward_no_grad Python frame
  v7_autograd_impl_sc   : + autograd_impl frame (TLS-shortcut: no predicate recompute)
  v8_autograd_impl_full : + predicate recompute (== Richard's prototype)

Each entry shares the same prelude: torch.compiler.is_compiling, kwargs guard,
_C._custom_op_fast_path_check, device/disabled/fn lookup. Differences are only
in the post-prelude no-grad body.

Run on a built PyTorch with reduce-custom_op-overhead's _C._custom_op_fast_path_check.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import statistics
import threading
import time

import torch
import torch.library
from torch import _C, _ops
from torch._library import autograd as _autograd_mod
from torch._library import custom_ops as _custom_ops_mod
from torch._library import utils as _lib_utils


# --------- Richard's _enable_fast_dispatch (monkey-patch onto OpOverload) ---------
_fast_dispatch_tls = threading.local()


@contextlib.contextmanager
def _enable_fast_dispatch(op, chain):
    prev = getattr(_fast_dispatch_tls, "entry", None)
    _fast_dispatch_tls.entry = (op, chain)
    try:
        yield
    finally:
        _fast_dispatch_tls.entry = prev


_orig_redispatch = _ops.OpOverload.redispatch


def _patched_redispatch(self, keyset, *args, **kwargs):
    entry = getattr(_fast_dispatch_tls, "entry", None)
    if entry is not None:
        target_op, chain = entry
        if target_op is self and chain:
            fn = chain.pop()
            return fn(keyset, *args, **kwargs)
    return _orig_redispatch(self, keyset, *args, **kwargs)


_ops.OpOverload.redispatch = _patched_redispatch


# --------- TLS for the "skip predicate recompute" shortcut ---------
_grad_shortcut_tls = threading.local()


# --------- ops ---------
@torch.library.custom_op("bench::nonmut", mutates_args=())
def nonmut(x: torch.Tensor) -> torch.Tensor:
    return x + 1


@nonmut.register_fake
def _(x):
    return torch.empty_like(x)


@nonmut.register_autograd
def _(ctx, grad):
    return grad


# --------- Variant builders ---------
# Each builder returns a `fast_call(*args, **kwargs)` to install on the opdef.
# All builders share the same prelude. Only the no-grad path differs.

_FALLBACK = _custom_ops_mod._FAST_PATH_FALLBACK


def _device_keyset(device_type: str):
    key_name = _C._dispatch_key_for_device(device_type)
    return _C.DispatchKeySet(getattr(_C.DispatchKey, key_name))


def make_variant(opdef: torch.library.CustomOpDef, mode: str):
    """Build a fast_call closure for the given mode. `mode` matches the
    variant table in the docstring."""
    op = opdef._opoverload
    op_name = opdef._name
    schema = op._schema
    raw_fns = opdef._raw_fns
    disabled = opdef._disabled_kernel
    is_mutable = schema.is_mutable
    if is_mutable:
        raise NotImplementedError("only nonmut here")
    keyset_for_device = {}

    def get_keyset(device_type):
        ks = keyset_for_device.get(device_type)
        if ks is None:
            ks = _device_keyset(device_type)
            keyset_for_device[device_type] = ks
        return ks

    after_autograd = _C._after_autograd_keyset

    # ---------- variant primitives ----------
    def aliasing(args, result):
        _lib_utils._c_check_aliasing_constraint(op_name, args, {}, result)

    def backend_dispatch(keyset, *args, **kwargs):
        device_type = args[0].device.type
        fn = raw_fns.get(device_type) or raw_fns.get(None)
        result = fn(*args, **kwargs)
        aliasing(args, result)
        return result

    @dataclasses.dataclass
    class Metadata:
        keyset: object
        keyword_only_args: dict

    # ----- no-grad bodies -----
    def body_v0_baseline(args, fn, _device, _any_grad):
        return fn(*args)

    def body_v1_alias(args, fn, _device, _any_grad):
        result = fn(*args)
        aliasing(args, result)
        return result

    def body_v2_backend_dispatch(args, fn, device, _any_grad):
        keyset = get_keyset(device)
        return backend_dispatch(keyset, *args)

    def body_v3_redispatch(args, fn, device, _any_grad):
        keyset = get_keyset(device)
        chain = [backend_dispatch]
        with _enable_fast_dispatch(op, chain):
            return op.redispatch(keyset, *args)

    def body_v4_autodisp(args, fn, device, _any_grad):
        keyset = get_keyset(device)
        chain = [backend_dispatch]
        with _enable_fast_dispatch(op, chain):
            with _C._AutoDispatchBelowAutograd():
                return op.redispatch(keyset & after_autograd, *args)

    def body_v5_metadata(args, fn, device, _any_grad):
        keyset = get_keyset(device)
        chain = [backend_dispatch]
        meta = Metadata(keyset, {})  # noqa: F841
        with _enable_fast_dispatch(op, chain):
            with _C._AutoDispatchBelowAutograd():
                return op.redispatch(keyset & after_autograd, *args)

    def forward_no_grad(*args):
        meta = args[-1]
        args = args[:-1]
        with _C._AutoDispatchBelowAutograd():
            return op.redispatch(meta.keyset & after_autograd, *args)

    def body_v6_forward_no_grad(args, fn, device, _any_grad):
        keyset = get_keyset(device)
        chain = [backend_dispatch]
        meta = Metadata(keyset, {})
        with _enable_fast_dispatch(op, chain):
            return forward_no_grad(*args, meta)

    def autograd_impl_sc(keyset, *args, **kwargs):
        # TLS-shortcut: read predicate from TLS (set by caller) instead of recomputing
        if getattr(_grad_shortcut_tls, "any_requires_grad", False):
            raise NotImplementedError("benchmark only does no-grad")
        return forward_no_grad(*args, Metadata(keyset, kwargs))

    def body_v7_autograd_impl_sc(args, fn, device, any_requires_grad):
        keyset = get_keyset(device)
        chain = [backend_dispatch]
        _grad_shortcut_tls.any_requires_grad = any_requires_grad
        with _enable_fast_dispatch(op, chain):
            return autograd_impl_sc(keyset, *args)

    def autograd_impl_full(keyset, *args, **kwargs):
        # Predicate recompute (richard's prototype reuses make_autograd_impl)
        if _C.is_grad_enabled() and _C._any_requires_grad(*args):
            raise NotImplementedError("benchmark only does no-grad")
        return forward_no_grad(*args, Metadata(keyset, kwargs))

    def body_v8_autograd_impl_full(args, fn, device, _any_grad):
        keyset = get_keyset(device)
        chain = [backend_dispatch]
        with _enable_fast_dispatch(op, chain):
            return autograd_impl_full(keyset, *args)

    bodies = {
        "v0_baseline": body_v0_baseline,
        "v1_alias": body_v1_alias,
        "v2_backend_dispatch": body_v2_backend_dispatch,
        "v3_redispatch": body_v3_redispatch,
        "v4_autodisp": body_v4_autodisp,
        "v5_metadata": body_v5_metadata,
        "v6_forward_no_grad": body_v6_forward_no_grad,
        "v7_autograd_impl_sc": body_v7_autograd_impl_sc,
        "v8_autograd_impl_full": body_v8_autograd_impl_full,
    }
    body = bodies[mode]

    def fast_call(*args, **kwargs):
        if torch.compiler.is_compiling():
            return _FALLBACK
        if not args or kwargs:
            return _FALLBACK

        check = _C._custom_op_fast_path_check(args)
        if check is None:
            return _FALLBACK

        device_type, any_requires_grad = check

        if device_type == "meta" or device_type in disabled:
            return _FALLBACK
        fn = raw_fns.get(device_type) or raw_fns.get(None)
        if fn is None:
            return _FALLBACK

        if torch.is_grad_enabled() and any_requires_grad:
            return _FALLBACK

        return body(args, fn, device_type, any_requires_grad)

    return fast_call


# --------- timing ---------
def timeit(fn, *, warmup=2000, n=20000, repeats=7):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        gc.disable()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        t1 = time.perf_counter_ns()
        gc.enable()
        ts.append((t1 - t0) / n)
    return min(ts), statistics.median(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0


def fmt(label, stats):
    mn, med, sd = stats
    return f"{label:<30}  min {mn:7.1f} ns   med {med:7.1f} ns   (sd {sd:5.1f})"


VARIANTS = [
    "v0_baseline",
    "v1_alias",
    "v2_backend_dispatch",
    "v3_redispatch",
    "v4_autodisp",
    "v5_metadata",
    "v6_forward_no_grad",
    "v7_autograd_impl_sc",
    "v8_autograd_impl_full",
]


def measure_real_current_pr(args_obj, n, warmup, repeats):
    """The original PR fast_call (untouched). Sanity check that v1_alias is close."""
    # Restore to whatever _install_fast_call set it to (already done).
    return timeit(lambda: nonmut(args_obj), warmup=warmup, n=n, repeats=repeats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1, help="Number of full passes over all variants; min across passes is reported.")
    parser.add_argument("--mode", default=None, help="Run only this variant and print min ns (use 'real' for the unmodified PR fast_call). Used by run_all.sh to isolate each variant in its own process.")
    args = parser.parse_args()

    print(f"# torch: {torch.__version__}")
    print(f"# repeats={args.repeats} n={args.n} warmup={args.warmup}")
    print()

    x = torch.randn(4)

    # Sanity check: confirm the C++ helper is present
    if not hasattr(_C, "_custom_op_fast_path_check"):
        raise SystemExit(
            "_C._custom_op_fast_path_check missing — must be built on reduce-custom_op-overhead"
        )

    if args.mode is not None:
        if args.mode == "real":
            s = timeit(lambda: nonmut(x), warmup=args.warmup, n=args.n, repeats=args.repeats)
        else:
            fast_call = make_variant(nonmut, args.mode)
            nonmut._fast_call = fast_call
            s = timeit(lambda: nonmut(x), warmup=args.warmup, n=args.n, repeats=args.repeats)
        print(fmt(args.mode, s))
        return

    # Reference: the real PR fast_call as installed by _install_fast_call.
    real_pr = nonmut._fast_call
    s_real = timeit(lambda: nonmut(x), warmup=args.warmup, n=args.n, repeats=args.repeats)
    print(fmt("real_current_pr_fast_call", s_real))

    import random

    rng = random.Random(args.shuffle_seed)
    results = {m: [] for m in VARIANTS}
    for p in range(args.passes):
        order = list(VARIANTS)
        if args.shuffle_seed:
            rng.shuffle(order)
        print(f"# pass {p + 1}/{args.passes} order: {order}")
        for mode in order:
            fast_call = make_variant(nonmut, mode)
            nonmut._fast_call = fast_call
            result = fast_call(x)
            if result is _FALLBACK:
                raise SystemExit(f"variant {mode}: fast_call returned FALLBACK")

            s = timeit(
                lambda: nonmut(x),
                warmup=args.warmup,
                n=args.n,
                repeats=args.repeats,
            )
            results[mode].append(s)

    # Aggregate: best (min-of-min) across passes
    agg = {}
    for mode in VARIANTS:
        mins = [s[0] for s in results[mode]]
        meds = [s[1] for s in results[mode]]
        agg[mode] = (min(mins), statistics.median(meds), max(mins) - min(mins))
        print(fmt(mode, agg[mode]))

    print()
    print("# Per-step deltas (best min across passes, ns):")
    prev = None
    for mode in VARIANTS:
        mn = agg[mode][0]
        if prev is None:
            print(f"  {mode:<30}  {mn:7.1f}    (kernel + fast_call prelude)")
        else:
            d = mn - prev
            print(f"  {mode:<30}  {mn:7.1f}   (+{d:7.1f})")
        prev = mn


if __name__ == "__main__":
    main()
