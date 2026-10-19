//! `matmul` `(m×k)·(k×n)` — the impl mirrored by lib-tensor's
//! `matmul.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

use super::codec::{Tensor, encode_tensor, require_tensor};
use super::elementwise::unbroadcast;
use super::shape_ops::transpose_dims;

/// The checked operand geometry of `matmul(a, b)`: ranks ∈ {2, 3}, the
/// inner dims match, and the batch dims broadcast (`1` stretches).
struct MatmulDims {
    /// Whether the output carries a batch axis (either side rank-3).
    batched: bool,
    /// Output batch extent (`max` of the two batch dims, `1` if unbatched).
    batch: usize,
    /// Per-side batch extents — `1` means that side repeats.
    b_a: usize,
    b_b: usize,
    m: usize,
    k: usize,
    n: usize,
}

/// Validate `a`/`b` for `matmul` and return the output geometry.
fn matmul_dims(a: &Tensor, b: &Tensor, op: &str) -> Result<MatmulDims, EvalError> {
    for (name, t) in [("lhs", a), ("rhs", b)] {
        if t.shape.len() != 2 && t.shape.len() != 3 {
            return Err(EvalError::Other(format!(
                "{op}: expected rank-2 or rank-3 {name}, got rank {} ({:?})",
                t.shape.len(),
                t.shape
            )));
        }
    }
    let batched = a.shape.len() == 3 || b.shape.len() == 3;
    let (b_a, m, k) = if a.shape.len() == 3 {
        (a.shape[0], a.shape[1], a.shape[2])
    } else {
        (1, a.shape[0], a.shape[1])
    };
    let (b_b, k2, n) = if b.shape.len() == 3 {
        (b.shape[0], b.shape[1], b.shape[2])
    } else {
        (1, b.shape[0], b.shape[1])
    };
    if k != k2 {
        return Err(EvalError::Other(format!(
            "{op}: inner dimension mismatch {k} vs {k2} ({m}x{k})·({k2}x{n})"
        )));
    }
    if b_a != b_b && b_a != 1 && b_b != 1 {
        return Err(EvalError::Other(format!(
            "{op}: batch dims {b_a} and {b_b} are not broadcast-compatible"
        )));
    }
    Ok(MatmulDims {
        batched,
        batch: b_a.max(b_b),
        b_a,
        b_b,
        m,
        k,
        n,
    })
}

/// The product itself, given validated [`MatmulDims`].
fn matmul_compute(a: &Tensor, b: &Tensor, d: &MatmulDims) -> Tensor {
    let (a_slab, b_slab) = (d.m * d.k, d.k * d.n);
    let mut data = vec![0.0; d.batch * d.m * d.n];
    for bi in 0..d.batch {
        let a_off = if d.b_a == 1 { 0 } else { bi * a_slab };
        let b_off = if d.b_b == 1 { 0 } else { bi * b_slab };
        let out_off = bi * d.m * d.n;
        for i in 0..d.m {
            for l in 0..d.k {
                let a_il = a.data[a_off + i * d.k + l];
                if a_il == 0.0 {
                    continue;
                }
                for j in 0..d.n {
                    data[out_off + i * d.n + j] += a_il * b.data[b_off + l * d.n + j];
                }
            }
        }
    }
    let shape = if d.batched {
        vec![d.batch, d.m, d.n]
    } else {
        vec![d.m, d.n]
    };
    Tensor { shape, data }
}

/// The Rust handl for `op "tensor.matmul" : (Tensor * Tensor) -> Tensor`.
///
/// Dense product `(m×k)·(k×n) -> m×n`, naive triple loop — the point of
/// this component is the composition seam, not BLAS. Rank-2 × rank-2 is
/// the plain matrix product; a rank-3 `[B, m, k]` or `[B, k, n]` side
/// batches it to `[B, m, n]`, with the batch dim broadcasting: a rank-2
/// operand (or a `B = 1` batch) repeats across the other's batches.
/// Ranks outside {2, 3} are an error.
#[derive(Debug, Clone, Copy)]
pub struct Matmul;
impl Function for Matmul {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        let dims = matmul_dims(&a, &b, "tensor.matmul")?;
        Ok(encode_tensor(matmul_compute(&a, &b, &dims)))
    }
}

/// The Rust handl for `op "tensor.matmul_backward" : (Tensor * Tensor * Tensor) -> Value`.
///
/// The adjoint [`DERIV_PAIRS`](crate::ops::DERIV_PAIRS) registers for
/// `tensor.matmul` — `matmul_backward(a, b, grad_out)`: `da = g·bᵀ`,
/// `db = aᵀ·g`, transposing the last two axes per batch. `grad_out` must
/// be the primal output's shape (`[B, m, n]` batched, `[m, n]` rank-2).
/// Each gradient is then unbroadcast over the batch dim to its operand's
/// shape — a rank-2 or `B = 1` operand's grad sums across the broadcast
/// batches. Returns `{"a": da, "b": db}`.
#[derive(Debug, Clone, Copy)]
pub struct MatmulBackward;
impl Function for MatmulBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let a = require_tensor(args, 0)?;
        let b = require_tensor(args, 1)?;
        let g = require_tensor(args, 2)?;
        let d = matmul_dims(&a, &b, "tensor.matmul_backward")?;
        let want = if d.batched {
            vec![d.batch, d.m, d.n]
        } else {
            vec![d.m, d.n]
        };
        if g.shape != want {
            return Err(EvalError::Other(format!(
                "tensor.matmul_backward: grad_out shape {:?} != matmul output {want:?}",
                g.shape
            )));
        }
        let bt = transpose_dims(&b, b.shape.len() - 2, b.shape.len() - 1)?;
        let at = transpose_dims(&a, a.shape.len() - 2, a.shape.len() - 1)?;
        // g·bᵀ is [batch, m, k]; aᵀ·g is [batch, k, n] — both satisfy the
        // same batch-broadcast rule the primal used.
        let da_full = matmul_compute(&g, &bt, &matmul_dims(&g, &bt, "tensor.matmul_backward")?);
        let db_full = matmul_compute(&at, &g, &matmul_dims(&at, &g, "tensor.matmul_backward")?);
        Ok(json!({
            "a": encode_tensor(unbroadcast(&da_full, &a.shape)?),
            "b": encode_tensor(unbroadcast(&db_full, &b.shape)?),
        }))
    }
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::{call_fn, tensor};

    #[test]
    fn matmul_2x2() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let v = call_fn(
            &Matmul,
            &[
                tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 2], &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn matmul_rectangular() {
        // [[1,2,3],[4,5,6]] (2x3) · [[7,8],[9,10],[11,12]] (3x2) = [[58,64],[139,154]]
        let v = call_fn(
            &Matmul,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 2], &[58.0, 64.0, 139.0, 154.0]));
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch_and_bad_rank() {
        assert!(
            call_fn(
                &Matmul,
                &[tensor(&[2, 3], &[0.0; 6]), tensor(&[2, 3], &[0.0; 6])]
            )
            .is_err()
        );
        assert!(call_fn(&Matmul, &[tensor(&[4], &[0.0; 4]), tensor(&[4], &[0.0; 4])]).is_err());
        // Rank 4 is outside the {2, 3} surface.
        assert!(
            call_fn(
                &Matmul,
                &[tensor(&[1, 1, 1, 1], &[0.0]), tensor(&[2, 2], &[0.0; 4])]
            )
            .is_err()
        );
    }

    #[test]
    fn matmul_batched() {
        // B=2: batch 0 is [[1,0],[0,1]]·x = x; batch 1 is 2·I·x = 2x.
        let a = tensor(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let b = tensor(&[2, 2, 1], &[3.0, 4.0, 5.0, 6.0]);
        let v = call_fn(&Matmul, &[a, b]).unwrap();
        assert_eq!(v, tensor(&[2, 2, 1], &[3.0, 4.0, 10.0, 12.0]));
    }

    #[test]
    fn matmul_rank2_broadcasts_over_batch() {
        // [2,2]×[2,2,2]: the same [[1,2],[3,4]] multiplies both batches.
        let a = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = tensor(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let v = call_fn(&Matmul, &[a, b]).unwrap();
        assert_eq!(
            v,
            tensor(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0])
        );
        // And the other side: [2,2,2]×[2,1] applies each batch to the
        // shared column vector.
        let a = tensor(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let b = tensor(&[2, 1], &[3.0, 4.0]);
        let v = call_fn(&Matmul, &[a, b]).unwrap();
        assert_eq!(v, tensor(&[2, 2, 1], &[3.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn matmul_rejects_incompatible_batches() {
        // [2,·,·] × [3,·,·]: neither batch dim is 1 and they differ.
        assert!(
            call_fn(
                &Matmul,
                &[
                    tensor(&[2, 1, 1], &[1.0, 1.0]),
                    tensor(&[3, 1, 1], &[1.0, 1.0, 1.0])
                ]
            )
            .is_err()
        );
        // Inner dims still checked under batching.
        assert!(
            call_fn(
                &Matmul,
                &[tensor(&[1, 2, 3], &[0.0; 6]), tensor(&[1, 2, 2], &[0.0; 4])]
            )
            .is_err()
        );
    }

    // ── matmul_backward ───────────────────────────────────────────────

    #[test]
    fn matmul_backward_2x2() {
        // a=[[1,2],[3,4]], b=[[5,6],[7,8]], g=I: da = I·bᵀ = bᵀ,
        // db = aᵀ·I = aᵀ.
        let v = call_fn(
            &MatmulBackward,
            &[
                tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
                tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2, 2], &[5.0, 7.0, 6.0, 8.0]));
        assert_eq!(v["b"], tensor(&[2, 2], &[1.0, 3.0, 2.0, 4.0]));
    }

    #[test]
    fn matmul_backward_unbroadcasts_shared_operand() {
        // a [2,2] = [[1,2],[3,4]] broadcast over B=2; b = [I, 2I];
        // g = [2,2,2] all-ones.
        // da_full = g·bᵀ = [ones, 2·ones] → da = Σ = [[3,3],[3,3]].
        // db = aᵀ·ones per batch = [[4,4],[6,6]] twice.
        let v = call_fn(
            &MatmulBackward,
            &[
                tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
                tensor(&[2, 2, 2], &[1.0; 8]),
            ],
        )
        .unwrap();
        assert_eq!(v["a"], tensor(&[2, 2], &[3.0, 3.0, 3.0, 3.0]));
        assert_eq!(
            v["b"],
            tensor(&[2, 2, 2], &[4.0, 4.0, 6.0, 6.0, 4.0, 4.0, 6.0, 6.0])
        );
    }

    #[test]
    fn matmul_backward_rejects_bad_grad_shape() {
        assert!(
            call_fn(
                &MatmulBackward,
                &[
                    tensor(&[2, 2], &[0.0; 4]),
                    tensor(&[2, 2], &[0.0; 4]),
                    tensor(&[2, 3], &[0.0; 6]),
                ]
            )
            .is_err()
        );
        // Batched output needs a batched grad.
        assert!(
            call_fn(
                &MatmulBackward,
                &[
                    tensor(&[2, 2, 2], &[0.0; 8]),
                    tensor(&[2, 2, 2], &[0.0; 8]),
                    tensor(&[2, 2], &[0.0; 4]),
                ]
            )
            .is_err()
        );
        assert!(call_fn(&MatmulBackward, &[tensor(&[2, 2], &[0.0; 4])]).is_err());
    }
}

