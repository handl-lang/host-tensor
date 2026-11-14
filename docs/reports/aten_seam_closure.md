# ATen seam closure — handwritten kernels for the declared seams

Status: **landed** — 16 of the 21 declared seams now have real
handwritten kernels; 5 remained honest stubs at landing (**Ported=202,
Seam=5**; a follow-up closed `unique`/`nonzero_dim` — see below —
current output **Ported=204, Seam=3**).

Baseline: commit `a95a68c` (generator + 174 real ports + 21 seams).
This pass added the kernels in `src/ops/aten_kernels.rs` ("Seam
closures" section, ~line 3085 onward), wired them through
`src/bin/aten_port_gen.rs`'s `MAP` table, and un-`# seam:`'d the
implemented pins in `tools/aten_port_list.txt`.

## Generator change

`resolve()` previously feature-gated on *every* schema arg: a required
arg of an unsupported type (`str`, `ScalarType`, `Device`, `Layout`,
`Generator`, `Dimname`, `complex`, `...`) forced `Status::Unsupported`,
and optional unsupported args were silently dropped from the decode
plan. That gate now runs **only for non-Raw emitters**. `Emit::Raw`
drivers receive `args: &[Value]` verbatim, so `str equation`
(`einsum`) and `ScalarType dtype` (`to.dtype`) are read by the kernel
itself — the decoder can't mis-decode what it never sees, and optional
unsupported args are *not* dropped for raw drivers (the kernel decides
whether to honor or reject them, e.g. `searchsorted`'s `sorter`,
`quantile`'s `interpolation`).

## Closed seams (16)

| op | pin | kernel | semantics / subset boundary |
|----|-----|--------|------------------------------|
| `einsum` | `aten::einsum` | `k_einsum` (raw) | `lhs->rhs` and implicit equations, `...` batch dims (right-aligned broadcast, extent-1 dims stretch), ≥2 operands, direct sum-of-products. Implicit output follows the numpy rule — broadcast ellipsis dims first, then once-labels **alphabetical** (verified against np 2.x: `ij,jh` yields the `(A@B)ᵀ` layout, the known gotcha). Repeated label inside one operand → loud error (no diagonal extraction). `path` ignored (contraction-order hint only). |
| `tensordot` | `aten::tensordot` | `k_tensordot` | dim-list contraction: equal-length dim lists, negative dims wrapped, duplicates rejected, extents must match; free axes permuted → `mm2` → free-shape output. |
| `kron` | `aten::kron` | `k_kron` | equal-rank operands; per-axis extent `a[d]·b[d]`. |
| `cross` | `aten::cross` | `k_cross` | `dim=None` → dim 0 for rank-1, error otherwise (torch's size-3 inference not guessed); explicit dim must have extent 3; operands equal shape. |
| `norm` | `aten::norm.Scalar` | `k_norm` | p-norm over ALL elements → rank-0; `p=0` counts nonzeros, `±inf` → max/min `|x|`. The `dim`/matrix-ord variants stay uncovered (pin is the flat schema). |
| `renorm` | `aten::renorm` | `k_renorm` | per-slice p-norm rescale `min(1, maxnorm/‖·‖_p)` along `dim`; zero norm leaves slice unchanged. |
| `normalize` | `aten::linalg_vector_norm` | `k_normalize` | `F.normalize` composite: `x/max(‖x‖_ord(dim), 1e-12)`; `dim=None` norms all axes; output always `t.shape`. |
| `median` | `aten::median` | `k_median` (raw) | flat → lower median (torch returns an element, not the mean of middle two); `.dim` sibling covered by arity dispatch → `{values, indices}`. |
| `mode` | `aten::mode` | `k_mode` | `{values, indices}` along dim; ties → smallest first-occurrence index (documented choice). |
| `kthvalue` | `aten::kthvalue` | `k_kthvalue` | `{values, indices}`; k 1-based, validated against extent; same stable sort as `sort`/`topk`. |
| `quantile` | `aten::quantile` | `k_quantile` (raw) | `"linear"` interpolation only (others error); `q` tensor or scalar; `dim=None` flattens; output = `q.shape ++ kept dims`. |
| `searchsorted` | `aten::searchsorted.Tensor` | `k_searchsorted` (raw) | last-dim binary search; `self` tensor (outer dims broadcast, extent-1 stretches) or scalar; `right`/`side` select lower/upper bound; NaN sorts last; `sorter` arg rejected; `out_int32` accepted but indices stay f64. `.Scalar` sibling covered by the same driver. |
| `bincount` | `aten::bincount` | `k_bincount` | non-negative integer inputs only (loud error otherwise), optional `weights`, `minlength`; index >1e8 rejected. |
| `meshgrid` | `aten::meshgrid` | `k_meshgrid` | ij indexing, rank-1 inputs only → `Tensor[]` JSON array; N-D broadcasting inputs rejected loudly. |
| `corrcoef` | `aten::corrcoef` | `k_corrcoef` | rowvar semantics; rank-1 → rank-0 (1.0 or NaN), rank-2 → m×m correlation, constant rows → NaN. |
| `to` | `aten::to.dtype` | `k_to` (raw) | `"float64"`/`"double"`/`"f64"` → identity; tensor rhs (`to.other`-style) → same-dtype copy; every other dtype string errors loudly. `non_blocking`/`copy`/`memory_format` accepted+ignored (immutable wire always returns a fresh tensor). |

## Promotions (12) — trivial coverage by existing kernels

`max`, `min` (raw driver dispatches flat / `.dim` tuple / `.other`
elementwise), `count_nonzero`, `inner` (`k_dot` — flattens, a superset
of the 1-D schema), `dist`, `allclose`, `is_nonzero`, `msort`,
`nansum`, `nanmean`, `cdist` (rank-2 and strictly-batched rank-3;
batch-dim broadcasting is a documented boundary), `pdist` (condensed
pairs, rank-2/3).

## Remaining seams (3) — and why

| op | pin | reason it stays a seam |
|----|-----|------------------------|
| `empty` | `aten::empty` | Uninitialized memory has no honest wire representation — every wire tensor carries concrete `data`. A zero/garbage fill would fake semantics. |
| `rand` | `aten::rand` | RNG: no `Generator` state exists on the wire, and nondeterministic output breaks the verify-by-replay model this surface is built on. |
| `normal` | `aten::normal` | Same RNG rationale (all `normal.*` overloads share it). |

### Follow-up closure (post-landed)

`unique` (`aten::_unique`) and `nonzero_dim` (`aten::nonzero_static`)
were closed after this report: `_unique` is private-named but a pure
functional 2-tuple — `k_unique` returns the `{"values","indices"}`
record encoding (sorted ascending with NaN last, or first-occurrence
order under `sorted=False`; `return_inverse` is inert per the schema's
fixed 2-tuple return). `nonzero_static` is the sized variant of
`nonzero` — `k_nonzero_static` pads the `[nnz, rank]` index tensor to
`[size, rank]` rows with `fill_value` and errors when `size < nnz`.
`is_out_or_inplace` exempts `_unique` by name (other `_`-private
schemas stay excluded). Current generator output: **Ported=204,
Seam=3**.

Mutation (`*_` in-place) and `out=` variants remain excluded by the
generator's existing variant filter — the wire model is immutable.

## Verification

- `cargo run --bin aten_port_gen` → `207 entries — "Ported=204 Seam=3"`
- `cargo run --bin aten_port_gen -- --check` → OK (generated surfaces in lockstep)
- `cargo check --all-targets` → clean
- `cargo test -p handl-host-tensor` → 97 lib + integration tests, all pass
  - new kernel unit tests cover einsum (explicit/implicit/ellipsis/
    errors), tensordot, kron, cross, norms, tuple reductions,
    meshgrid, bincount, corrcoef, `to`, searchsorted, quantile, and
    every promotion;
  - `aten_generated_ports_smoke` in `src/ops/mod.rs` now exercises
    einsum/tensordot/to/meshgrid/max end-to-end and asserts the five
    remaining seams still fail loudly.

## Honest-accounting notes

- `einsum` is the naive O(out·contract·Σrank) engine — correct f64
  semantics, no `opt_einsum` path optimization (`path` arg ignored).
- `median.dim`'s *unstable* tie behavior and `mode`'s tie-break are
  documented choices, not bit-exact torch guarantees.
- `to` is f64-identity only by construction; the codec cannot represent
  other dtypes, so non-f64 requests error rather than silently degrade.
- `cdist` rank-3 requires equal batch dims (torch broadcasts them).
- `searchsorted` ignores `out_int32` (indices are always f64 on the
  wire) and rejects `sorter`.
- NaN cannot ride the JSON wire (`serde_json` has no NaN literal), so
  NaN-handling kernels (`nansum`, `nanmean`, `isnan` paths) are unit-tested
  at the `Tensor` level, not through `tensor()` wire fixtures.
