#!/usr/bin/env bash
# hnl gate — `typecheck` + `lint` + `test` over this package's `.hnl` sources.
#
# Sources live in `src/` (one package per repo, ADR-0128 split). Nothing else
# covers them — `cargo` never sees `.hnl`.
#
# Usage:  HNL=../host-cli/target/debug/hnl scripts/hnl-gate.sh
set -euo pipefail

SRC_DIR="${SRC_DIR:-src}"
HNL="${HNL:-../host-cli/target/debug/hnl}"
if [ ! -x "$HNL" ]; then
  echo "error: no hnl binary at '$HNL'" >&2
  echo "       build it:   (cd ../host-cli && cargo build --bin hnl)" >&2
  exit 1
fi

# The corpus is mixed-dialect: canonical `.hnl` and arrow/contract `.hnl`
# (`~>`, `cap`/`op`/`handl`/`tag` declarations, a leading `axiom`). The
# canonical tools cannot parse the arrow dialect, so mirror
# `scripts/dogfood-gate.sh`'s detector and gate only the canonical files.
is_canonical() {
  awk '
  function trim(s) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); return s }
  function strip_comment(s) { gsub(/--.*$/, "", s); return trim(s) }
  function top_level(s) {
    if (s ~ /^[[:space:]]*$/) return ""
    if (s ~ /^[[:space:]]/) return ""
    return strip_comment(s)
  }
  BEGIN { first=1; arrow=0 }
  {
    tl = top_level($0)
    if (first && tl != "") {
      if (tl ~ /^axiom /) { exit 1 }
      first = 0
    }
    if (tl ~ /^(cap|op|handl|tag) /) { arrow = 1 }
    code = strip_comment($0)
    if (code != "") {
      if (index(code, "|||") || code ~ /(^|[^a-zA-Z0-9_.])(arr|first|choice|loop|comp|app)\(/)
        { arrow = 1 }
      if (code ~ /(^|[^a-zA-Z0-9_.])~>/) { arrow = 1 }
      if (code ~ /(^|[^a-zA-Z0-9_.])rec[[:space:]]*\{/) { arrow = 1 }
    }
  }
  END { exit (arrow ? 1 : 0) }
  ' "$1"
}

sources=()
tests=()
skipped=0
while IFS= read -r f; do
  if is_canonical "$f"; then
    case "$f" in
      *.test.hnl) tests+=("$f") ;;
      *) sources+=("$f") ;;
    esac
  else
    skipped=$((skipped + 1))
  fi
done < <(find "$SRC_DIR" -name '*.hnl' | sort)

if [ "${#sources[@]}" -eq 0 ] && [ "${#tests[@]}" -eq 0 ]; then
  echo "only non-canonical (arrow/contract) sources under $SRC_DIR/ — gate skipped"
  exit 0
fi

echo "== hnl typecheck (${#sources[@]} sources, ${#tests[@]} test files, ${skipped} dialect skipped) =="
"$HNL" typecheck "${sources[@]}" ${tests[@]+"${tests[@]}"}

echo
echo "== hnl lint =="
"$HNL" lint "${sources[@]}" ${tests[@]+"${tests[@]}"}

if [ "${#tests[@]}" -gt 0 ]; then
  echo
  # `--jobs 1` inside each file: these suites can share one fixture process,
  # so running a file's cases concurrently makes them fight over it.
  echo "== hnl test (per file; in-file jobs: 1) =="
  failed_files=0
  for t in "${tests[@]}"; do
    echo "-- $t"
    if out=$("$HNL" test --jobs 1 "$t" 2>&1); then
      printf '%s\n' "$out" | tail -2
      continue
    fi
    printf '%s\n' "$out"
    # A package repo ships no host module chain, so cases that reach a native
    # it does not provide (`uchar.*`, `unorm2.*`, …) fail with
    # `call_host: function '<name>' not found`. Those are host-dependent by
    # construction: report them, but do not fail the gate on them alone.
    real=$(printf '%s\n' "$out" | grep '^FAIL' | grep -vc 'call_host: function ' || true)
    if [ "${real:-0}" -eq 0 ]; then
      echo "   (all failures are missing host natives — not counted)"
    else
      failed_files=$((failed_files + 1))
    fi
  done
  if [ "$failed_files" -ne 0 ]; then
    echo "hnl gate FAILED: $failed_files test file(s) with real failures." >&2
    exit 1
  fi
fi

echo
echo "hnl gate passed."
