//! `matmul` `(m×k)·(k×n)` — the impl mirrored by lib-tensor's
//! `matmul.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::Value;

use super::codec::{Tensor, encode_tensor, require_tensor};

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
        for (name, t) in [("lhs", &a), ("rhs", &b)] {
            if t.shape.len() != 2 && t.shape.len() != 3 {
                return Err(EvalError::Other(format!(
                    "tensor.matmul: expected rank-2 or rank-3 {name}, got rank {} ({:?})",
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
                "tensor.matmul: inner dimension mismatch {k} vs {k2} ({m}x{k})·({k2}x{n})"
            )));
        }
        if b_a != b_b && b_a != 1 && b_b != 1 {
            return Err(EvalError::Other(format!(
                "tensor.matmul: batch dims {b_a} and {b_b} are not broadcast-compatible"
            )));
        }
        let batch = b_a.max(b_b);
        let (a_slab, b_slab) = (m * k, k * n);
        let mut data = vec![0.0; batch * m * n];
        for bi in 0..batch {
            let a_off = if b_a == 1 { 0 } else { bi * a_slab };
            let b_off = if b_b == 1 { 0 } else { bi * b_slab };
            let out_off = bi * m * n;
            for i in 0..m {
                for l in 0..k {
                    let a_il = a.data[a_off + i * k + l];
                    if a_il == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        data[out_off + i * n + j] += a_il * b.data[b_off + l * n + j];
                    }
                }
            }
        }
        let shape = if batched {
            vec![batch, m, n]
        } else {
            vec![m, n]
        };
        Ok(encode_tensor(Tensor { shape, data }))
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
}
