#!/usr/bin/env bash
# Run each variant in its own python process (clean state) and collect min ns.
# Pinned to a single core. Multiple passes; final report is min over all passes.

set -e

cd "$(dirname "$0")"

PASSES=${PASSES:-5}
CORE=${CORE:-8}
REPEATS=${REPEATS:-7}
N=${N:-20000}
WARMUP=${WARMUP:-2000}

VARIANTS=(real v0_baseline v1_alias v2_backend_dispatch v3_redispatch v4_autodisp v5_metadata v6_forward_no_grad v7_autograd_impl_sc v8_autograd_impl_full)

declare -A BEST
for v in "${VARIANTS[@]}"; do BEST[$v]=99999999; done

echo "# isolated subprocess per variant; passes=$PASSES core=$CORE repeats=$REPEATS n=$N"
echo

for ((p=1; p<=PASSES; p++)); do
  echo "## pass $p"
  # Shuffle the order across passes
  ORDER=( $(printf '%s\n' "${VARIANTS[@]}" | shuf --random-source=/dev/urandom) )
  for v in "${ORDER[@]}"; do
    LINE=$(taskset -c $CORE python profile_redispatch_chain.py --mode $v --repeats $REPEATS --n $N --warmup $WARMUP 2>/dev/null | grep -E "^${v//+/\\+}\b|^real\b" | tail -1)
    MIN=$(echo "$LINE" | awk '{print $3}')
    if [ -n "$MIN" ]; then
      MIN_INT=$(printf '%.0f' "$MIN")
      cur=${BEST[$v]}
      if [ "$MIN_INT" -lt "$cur" ]; then
        BEST[$v]=$MIN_INT
      fi
      printf "  %-30s  %7d ns\n" "$v" "$MIN_INT"
    else
      echo "  $v: NO RESULT (line='$LINE')"
    fi
  done
  echo
done

echo "## Best min across all passes"
prev=""
for v in "${VARIANTS[@]}"; do
  cur=${BEST[$v]}
  if [ -z "$prev" ]; then
    printf "  %-30s  %7d ns    (kernel + fast_call prelude)\n" "$v" "$cur"
  else
    d=$((cur - prev))
    if [ "$d" -ge 0 ]; then
      printf "  %-30s  %7d ns   (+%6d)\n" "$v" "$cur" "$d"
    else
      printf "  %-30s  %7d ns   (%7d)\n" "$v" "$cur" "$d"
    fi
  fi
  prev=$cur
done
