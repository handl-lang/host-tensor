//! `reshape` / `transpose` / `concat`: the re-labeling ops — the impl
//! mirrored by lib-tensor's `shape_ops.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::Value;

use super::codec::{
    Tensor, decode_tensor, encode_tensor, require_extent, require_shape, require_tensor,
    type_mismatch,
};

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

/// Row-major strides of `shape` (the contiguous layout every tensor in
/// this component uses).
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

/// The Rust handl for `op "tensor.reshape" : (Tensor * (List Int)) -> Tensor`.
///
/// Re-labels the flat row-major payload under a new shape; the extents'
/// product must equal the element count (the data order never changes —
/// a reshape is a view, not a permutation). No `-1` wildcard: the full
/// shape is spelled out, matching `zeros`' shape convention.
#[derive(Debug, Clone, Copy)]
pub struct Reshape;
impl Function for Reshape {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let t = require_tensor(args, 0)?;
        let shape = require_shape(args, 1)?;
        let expected = shape.iter().product::<usize>();
        if expected != t.data.len() {
            return Err(EvalError::Other(format!(
                "tensor.reshape: shape {shape:?} has {expected} elements, \
                 tensor {:?} has {}",
                t.shape,
                t.data.len()
            )));
        }
        Ok(encode_tensor(Tensor {
            shape,
            data: t.data,
        }))
    }
}

/// The Rust handl for `op "tensor.transpose" : (Tensor * Int * Int) -> Tensor`.
///
/// Swaps two axes: `transpose(t, d0, d1)` returns the tensor whose
/// `(…, i_d1, …, i_d0, …)` element is `t`'s `(…, i_d0, …, i_d1, …)`.
/// `d0 == d1` is the identity. A rank-2 `transpose(t, 0, 1)` is the
/// matrix transpose.
#[derive(Debug, Clone, Copy)]
pub struct Transpose;
impl Function for Transpose {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 3)?;
        let t = require_tensor(args, 0)?;
        let rank = t.shape.len();
        let d0 = require_extent(&args[1])?;
        let d1 = require_extent(&args[2])?;
        if d0 >= rank || d1 >= rank {
            return Err(EvalError::Other(format!(
                "tensor.transpose: dims ({d0}, {d1}) out of range for rank {rank} ({:?})",
                t.shape
            )));
        }
        // `perm[d]` = the input axis output axis `d` reads from.
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(d0, d1);
        let out_shape: Vec<usize> = perm.iter().map(|&d| t.shape[d]).collect();
        let in_strides = row_major_strides(&t.shape);
        // Output odometer → input offset via the permuted strides.
        let mut data = Vec::with_capacity(t.data.len());
        let mut idx = vec![0usize; rank];
        for _ in 0..t.data.len() {
            let off: usize = idx
                .iter()
                .zip(&perm)
                .map(|(&i, &d)| i * in_strides[d])
                .sum();
            data.push(t.data[off]);
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < out_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(encode_tensor(Tensor {
            shape: out_shape,
            data,
        }))
    }
}

/// The Rust handl for `op "tensor.concat" : (Array * Int) -> Tensor`.
///
/// `concat(xs, axis)` joins a non-empty list of tensors along `axis`.
/// All inputs must share a rank and agree on every axis but `axis`,
/// whose extents sum. `concat(xs, 0)` stacks, `concat(xs, rank−1)`
/// appends along the fastest-varying axis.
#[derive(Debug, Clone, Copy)]
pub struct Concat;
impl Function for Concat {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 2)?;
        let arr = args[0]
            .as_array()
            .ok_or_else(|| type_mismatch("tensor.concat: array of tensors", "non-array"))?;
        if arr.is_empty() {
            return Err(EvalError::Other(
                "tensor.concat: expected a non-empty list of tensors".to_string(),
            ));
        }
        let axis = require_extent(&args[1])?;
        let ts: Vec<Tensor> = arr.iter().map(decode_tensor).collect::<Result<_, _>>()?;
        let rank = ts[0].shape.len();
        if rank == 0 {
            return Err(EvalError::Other(
                "tensor.concat: cannot concat rank-0 tensors".to_string(),
            ));
        }
        if axis >= rank {
            return Err(EvalError::Other(format!(
                "tensor.concat: axis {axis} out of range for rank {rank} ({:?})",
                ts[0].shape
            )));
        }
        for (i, t) in ts.iter().enumerate().skip(1) {
            if t.shape.len() != rank {
                return Err(EvalError::Other(format!(
                    "tensor.concat: element {i} has rank {} != {rank} ({:?} vs {:?})",
                    t.shape.len(),
                    t.shape,
                    ts[0].shape
                )));
            }
            for (d, (&a, &b)) in t.shape.iter().zip(ts[0].shape.iter()).enumerate() {
                if d != axis && a != b {
                    return Err(EvalError::Other(format!(
                        "tensor.concat: element {i} differs on non-concat axis {d} \
                         ({:?} vs {:?})",
                        t.shape, ts[0].shape
                    )));
                }
            }
        }
        // Row-major concat along `axis`: dims before `axis` are equal
        // across inputs (the `outer` repeat count), dims after are equal
        // (the `inner` contiguous run per axis element). Each input's
        // per-outer block is contiguous — copy block-by-block.
        let outer: usize = ts[0].shape[..axis].iter().product();
        let inner: usize = ts[0].shape[axis + 1..].iter().product();
        let mut out_shape = ts[0].shape.clone();
        out_shape[axis] = ts.iter().map(|t| t.shape[axis]).sum();
        let mut data = Vec::with_capacity(out_shape.iter().product());
        for o in 0..outer {
            for t in &ts {
                let slab = t.shape[axis] * inner;
                data.extend_from_slice(&t.data[o * slab..(o + 1) * slab]);
            }
        }
        Ok(encode_tensor(Tensor {
            shape: out_shape,
            data,
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
    use serde_json::json;

    #[test]
    fn reshape_relables_shape_keeps_data() {
        let v = call_fn(
            &Reshape,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!([3, 2]),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        // To rank-1 and to scalar.
        assert_eq!(
            call_fn(&Reshape, &[tensor(&[2, 2], &[1.0; 4]), json!([4])]).unwrap(),
            tensor(&[4], &[1.0; 4])
        );
        assert_eq!(
            call_fn(&Reshape, &[tensor(&[1], &[7.0]), json!([])]).unwrap(),
            tensor(&[], &[7.0])
        );
    }

    #[test]
    fn reshape_rejects_wrong_element_count() {
        assert!(call_fn(&Reshape, &[tensor(&[2, 3], &[0.0; 6]), json!([2, 2])]).is_err());
        assert!(call_fn(&Reshape, &[tensor(&[2, 3], &[0.0; 6]), json!([7])]).is_err());
    }

    #[test]
    fn transpose_matrix() {
        // [[1,2,3],[4,5,6]]ᵀ = [[1,4],[2,5],[3,6]].
        let v = call_fn(
            &Transpose,
            &[
                tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!(0),
                json!(1),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn transpose_rank3_swaps_inner_axes() {
        // [1,2,3] = [[[1,2,3],[4,5,6]]] → transpose(1,2) = [1,3,2].
        let v = call_fn(
            &Transpose,
            &[
                tensor(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!(1),
                json!(2),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[1, 3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
        // transpose(0,2): [1,2,3] → [3,2,1]; out[i,j,0] = in[0,j,i], so
        // out = [[[1],[4]], [[2],[5]], [[3],[6]]] flattened.
        let v = call_fn(
            &Transpose,
            &[
                tensor(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                json!(0),
                json!(2),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2, 1], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]));
    }

    #[test]
    fn transpose_identity_and_rejects_bad_dims() {
        let t = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            call_fn(&Transpose, &[t.clone(), json!(1), json!(1)]).unwrap(),
            t
        );
        assert!(call_fn(&Transpose, &[t.clone(), json!(0), json!(2)]).is_err());
        assert!(call_fn(&Transpose, &[t.clone(), json!(5), json!(0)]).is_err());
        assert!(call_fn(&Transpose, &[tensor(&[], &[1.0]), json!(0), json!(1)]).is_err());
    }

    #[test]
    fn concat_axis0_stacks() {
        // [[1,2],[3,4]] ++ [[5,6]] along axis 0 → [[1,2],[3,4],[5,6]].
        let v = call_fn(
            &Concat,
            &[
                json!([
                    tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                    tensor(&[1, 2], &[5.0, 6.0])
                ]),
                json!(0),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn concat_axis1_appends_columns() {
        // [[1,2],[3,4]] ++ [[5],[6]] along axis 1 → [[1,2,5],[3,4,6]].
        let v = call_fn(
            &Concat,
            &[
                json!([
                    tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                    tensor(&[2, 1], &[5.0, 6.0])
                ]),
                json!(1),
            ],
        )
        .unwrap();
        assert_eq!(v, tensor(&[2, 3], &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]));
    }

    #[test]
    fn concat_rejects_bad_inputs() {
        let t = tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        // empty list / non-array / non-tensor element.
        assert!(call_fn(&Concat, &[json!([]), json!(0)]).is_err());
        assert!(call_fn(&Concat, &[json!(5), json!(0)]).is_err());
        assert!(call_fn(&Concat, &[json!([t.clone(), json!(7)]), json!(0)]).is_err());
        // axis out of range / mismatched non-concat axis / rank mismatch.
        assert!(call_fn(&Concat, &[json!([t.clone()]), json!(2)]).is_err());
        assert!(
            call_fn(
                &Concat,
                &[json!([t.clone(), tensor(&[2, 3], &[0.0; 6])]), json!(0)]
            )
            .is_err()
        );
        assert!(
            call_fn(
                &Concat,
                &[json!([t.clone(), tensor(&[4], &[0.0; 4])]), json!(0)]
            )
            .is_err()
        );
        // rank-0 tensors.
        assert!(
            call_fn(
                &Concat,
                &[json!([tensor(&[], &[1.0]), tensor(&[], &[2.0])]), json!(0)]
            )
            .is_err()
        );
    }
}
