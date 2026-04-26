"""Compute medians/mins of per-step deltas across all runs in result_5runs_in_order.txt."""
import re
import statistics
import sys

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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "result_5runs_in_order.txt"
    text = open(path).read()
    runs = re.split(r"^=== run \d+ ===\s*$", text, flags=re.MULTILINE)
    runs = [r for r in runs if r.strip()]
    print(f"# {len(runs)} runs found")

    by_variant = {v: [] for v in VARIANTS}
    for r in runs:
        for v in VARIANTS:
            m = re.search(rf"^{v}\s+min\s+([\d.]+)\s+ns", r, flags=re.MULTILINE)
            if m:
                by_variant[v].append(float(m.group(1)))

    print(f"# samples per variant: {[(v, len(by_variant[v])) for v in VARIANTS]}\n")

    # Per-step deltas, computed within each run, then aggregated
    deltas = {f"{a}->{b}": [] for a, b in zip(VARIANTS, VARIANTS[1:])}
    for r in runs:
        vals = {}
        for v in VARIANTS:
            m = re.search(rf"^{v}\s+min\s+([\d.]+)\s+ns", r, flags=re.MULTILINE)
            if m:
                vals[v] = float(m.group(1))
        if len(vals) != len(VARIANTS):
            continue
        for a, b in zip(VARIANTS, VARIANTS[1:]):
            deltas[f"{a}->{b}"].append(vals[b] - vals[a])

    print(f"{'Step':<40}  {'median':>8}  {'min':>8}  {'max':>8}  {'IQR':>8}")
    total_med = 0
    for k in deltas:
        d = deltas[k]
        if not d:
            continue
        med = statistics.median(d)
        q = statistics.quantiles(d, n=4) if len(d) >= 4 else [min(d), med, max(d)]
        iqr = q[2] - q[0]
        print(f"{k:<40}  {med:8.0f}  {min(d):8.0f}  {max(d):8.0f}  {iqr:8.0f}")
        total_med += med

    print()
    print(f"sum of medians (v0->v8): {total_med:.0f} ns")
    if "v0_baseline->v1_alias" in deltas:
        gap_med = total_med - statistics.median(deltas["v0_baseline->v1_alias"])
        print(f"sum of medians excluding v0->v1 (the alias check that current PR also pays): {gap_med:.0f} ns")


if __name__ == "__main__":
    main()
