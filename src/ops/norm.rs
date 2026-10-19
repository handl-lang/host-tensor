//! `batch_norm` / `batch_norm_backward`: eval-mode normalization and its
//! registered adjoint — the impl mirrored by lib-tensor's `norm.hnl`.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

use super::codec::{
    Tensor, encode_tensor, opt_nonneg_float, require_arg_range, require_tensor,
};

/// The Rust handl for `op "tensor.batch_norm" : Value... -> Tensor`.
///
/// Eval-mode (inference) batch normalization:
/// `batch_norm(input, gamma, beta, mean, var, eps?)` with `input` rank-4
/// `[N, C, H, W]` and `gamma` / `beta` / `mean` / `var` rank-1 `[C]`
/// learned/statistical parameters. Per channel `c`,
/// `out = (x − mean[c]) / sqrt(var[c] + eps) · gamma[c] + beta[c]`.
/// `eps` is `null`/absent (→ `1e-5`) or a non-negative finite number.
#[derive(Debug, Clone, Copy)]
pub struct BatchNorm;
impl Function for BatchNorm {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 5, 6, "tensor.batch_norm")?;
        let input = require_tensor(args, 0)?;
        if input.shape.len() != 4 {
            return Err(EvalError::Other(format!(
                "tensor.batch_norm: expected a rank-4 input [N, C, H, W], got rank {} ({:?})",
                input.shape.len(),
                input.shape
            )));
        }
        let c = input.shape[1];
        let params: Vec<Tensor> = (1..5)
            .map(|pos| require_tensor(args, pos))
            .collect::<Result<_, _>>()?;
        let names = ["gamma", "beta", "mean", "var"];
        for (name, p) in names.iter().zip(params.iter()) {
            if p.shape != [c] {
                return Err(EvalError::Other(format!(
                    "tensor.batch_norm: {name} shape {:?} != [{c}]",
                    p.shape
                )));
            }
        }
        let [gamma, beta, mean, var] = [&params[0], &params[1], &params[2], &params[3]];
        let eps = opt_nonneg_float(args, 5, 1e-5, "tensor.batch_norm: eps")?;
        let hw = input.shape[2] * input.shape[3];
        let mut data = vec![0.0; input.data.len()];
        for bn in 0..input.shape[0] {
            for ch in 0..c {
                let inv_std = (var.data[ch] + eps).sqrt().recip();
                let g = gamma.data[ch] * inv_std;
                let shift = beta.data[ch] - mean.data[ch] * g;
                let base = (bn * c + ch) * hw;
                for i in 0..hw {
                    data[base + i] = input.data[base + i] * g + shift;
                }
            }
        }
        Ok(encode_tensor(Tensor {
            shape: input.shape,
            data,
        }))
    }
}

// ---------------------------------------------------------------------------
// Derivatives
// ---------------------------------------------------------------------------

/// The Rust handl for `op "tensor.batch_norm_backward" : Value... -> Value`.
///
/// The adjoint [`DERIV_PAIRS`] registers for `tensor.batch_norm` — what
/// `tensor.derive("batch_norm")` resolves to. The signature mirrors the
/// primal plus the upstream gradient: `batch_norm_backward(input,
/// gamma, beta, mean, var, grad_out, eps?)`. It returns an object of
/// the gradients of the primal's *differentiable* inputs —
/// `{"input": dx, "gamma": dγ, "beta": dβ}` — `mean`/`var` are
/// statistical in eval mode and carry no grad. Per channel `c`, with
/// `x̂ = (x − μ_c)/√(σ²_c + ε)` and upstream grad `g`:
/// `dβ_c = Σ g`, `dγ_c = Σ g·x̂`, `dx = g·γ_c·inv_std_c` (sums over the
/// batch and spatial dims).
#[derive(Debug, Clone, Copy)]
pub struct BatchNormBackward;
impl Function for BatchNormBackward {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        require_arg_range(args, 6, 7, "tensor.batch_norm_backward")?;
        let input = require_tensor(args, 0)?;
        if input.shape.len() != 4 {
            return Err(EvalError::Other(format!(
                "tensor.batch_norm_backward: expected a rank-4 input [N, C, H, W], \
                 got rank {} ({:?})",
                input.shape.len(),
                input.shape
            )));
        }
        let c = input.shape[1];
        let params: Vec<Tensor> = (1..5)
            .map(|pos| require_tensor(args, pos))
            .collect::<Result<_, _>>()?;
        let names = ["gamma", "beta", "mean", "var"];
        for (name, p) in names.iter().zip(params.iter()) {
            if p.shape != [c] {
                return Err(EvalError::Other(format!(
                    "tensor.batch_norm_backward: {name} shape {:?} != [{c}]",
                    p.shape
                )));
            }
        }
        let [gamma, _beta, mean, var] = [&params[0], &params[1], &params[2], &params[3]];
        let grad = require_tensor(args, 5)?;
        if grad.shape != input.shape {
            return Err(EvalError::Other(format!(
                "tensor.batch_norm_backward: grad_out shape {:?} != input shape {:?}",
                grad.shape, input.shape
            )));
        }
        let eps = opt_nonneg_float(args, 6, 1e-5, "tensor.batch_norm_backward: eps")?;
        let hw = input.shape[2] * input.shape[3];
        let mut dx = vec![0.0; input.data.len()];
        let mut dgamma = vec![0.0; c];
        let mut dbeta = vec![0.0; c];
        for bn in 0..input.shape[0] {
            for ch in 0..c {
                let inv_std = (var.data[ch] + eps).sqrt().recip();
                let scale = gamma.data[ch] * inv_std;
                let base = (bn * c + ch) * hw;
                for i in 0..hw {
                    let g = grad.data[base + i];
                    let x_hat = (input.data[base + i] - mean.data[ch]) * inv_std;
                    dbeta[ch] += g;
                    dgamma[ch] += g * x_hat;
                    dx[base + i] = g * scale;
                }
            }
        }
        Ok(json!({
            "input": encode_tensor(Tensor { shape: input.shape, data: dx }),
            "gamma": encode_tensor(Tensor { shape: vec![c], data: dgamma }),
            "beta": encode_tensor(Tensor { shape: vec![c], data: dbeta }),
        }))
    }
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::codec::decode_tensor;
    use crate::ops::testutil::{call_fn, tensor};
    use serde_json::json;

    #[test]
    fn batch_norm_normalizes_per_channel() {
        // input [1,2,1,2] = [[1,2],[3,4]]; eps = 0 for exact math.
        // c0: (x−0)/√1 ·1 +0 = x; c1: (x−3)/√4 ·2 +1 = x−2.
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let args = [
            input,
            tensor(&[2], &[1.0, 2.0]), // gamma
            tensor(&[2], &[0.0, 1.0]), // beta
            tensor(&[2], &[0.0, 3.0]), // mean
            tensor(&[2], &[1.0, 4.0]), // var
            json!(0),
        ];
        let v = call_fn(&BatchNorm, &args).unwrap();
        assert_eq!(v, tensor(&[1, 2, 1, 2], &[1.0, 2.0, 1.0, 2.0]));
    }

    #[test]
    fn batch_norm_default_eps_and_batch_dim() {
        // N=2, C=1: x = (x − 5)/√(4 + 1e-5).
        let input = tensor(&[2, 1, 1, 1], &[3.0, 7.0]);
        let v = call_fn(
            &BatchNorm,
            &[
                input,
                tensor(&[1], &[1.0]),
                tensor(&[1], &[0.0]),
                tensor(&[1], &[5.0]),
                tensor(&[1], &[4.0]),
            ],
        )
        .unwrap();
        let t = decode_tensor(&v).unwrap();
        let inv = (4.0f64 + 1e-5).sqrt().recip();
        assert!((t.data[0] - (3.0 - 5.0) * inv).abs() < 1e-12);
        assert!((t.data[1] - (7.0 - 5.0) * inv).abs() < 1e-12);
    }

    #[test]
    fn batch_norm_rejects_bad_shapes_and_eps() {
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let good = |p: &[f64]| tensor(&[2], p);
        // non-rank-4 input.
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    tensor(&[4], &[0.0; 4]),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                ]
            )
            .is_err()
        );
        // param wrong length (gamma [3] vs C=2).
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input.clone(),
                    tensor(&[3], &[1.0; 3]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                ]
            )
            .is_err()
        );
        // negative eps.
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input.clone(),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                    json!(-1),
                ]
            )
            .is_err()
        );
        // arity: missing var / extra arg.
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input.clone(),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2])
                ]
            )
            .is_err()
        );
        assert!(
            call_fn(
                &BatchNorm,
                &[
                    input,
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                    json!(0),
                    json!(0),
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn batch_norm_backward_eval_mode() {
        // input [1,2,1,2] = [[1,2],[3,4]]; eps = 0.
        // c0: μ=0, σ²=1, γ=1 → x̂ = x; dx = g·1·1 = g.
        // c1: μ=3, σ²=4, γ=2 → inv_std = 1/2; dx = g·2·1/2 = g.
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let grad = tensor(&[1, 2, 1, 2], &[0.5, 1.0, 1.5, 2.0]);
        let v = call_fn(
            &BatchNormBackward,
            &[
                input,
                tensor(&[2], &[1.0, 2.0]), // gamma
                tensor(&[2], &[0.0, 1.0]), // beta (present, unused)
                tensor(&[2], &[0.0, 3.0]), // mean
                tensor(&[2], &[1.0, 4.0]), // var
                grad,
                json!(0),
            ],
        )
        .unwrap();
        // dx = g elementwise (both channels' scales are 1 under this setup).
        assert_eq!(v["input"], tensor(&[1, 2, 1, 2], &[0.5, 1.0, 1.5, 2.0]));
        // dβ_c = Σ g over (n, hw): c0 → 1.5, c1 → 3.5.
        assert_eq!(v["beta"], tensor(&[2], &[1.5, 3.5]));
        // dγ_c = Σ g·x̂: c0 → 0.5·1 + 1·2 = 2.5; c1 → 1.5·0 + 2·0.5 = 1.0
        // (x̂ = (x−3)/2 = {0, 0.5}).
        assert_eq!(v["gamma"], tensor(&[2], &[2.5, 1.0]));
    }

    #[test]
    fn batch_norm_backward_sums_over_batch() {
        // N=2, C=1: Σ over both batch elements.
        let input = tensor(&[2, 1, 1, 1], &[1.0, 3.0]);
        let grad = tensor(&[2, 1, 1, 1], &[2.0, 4.0]);
        let v = call_fn(
            &BatchNormBackward,
            &[
                input,
                tensor(&[1], &[1.0]), // gamma
                tensor(&[1], &[0.0]), // beta
                tensor(&[1], &[0.0]), // mean
                tensor(&[1], &[1.0]), // var
                grad,
                json!(0),
            ],
        )
        .unwrap();
        assert_eq!(v["beta"], tensor(&[1], &[6.0]));
        assert_eq!(v["gamma"], tensor(&[1], &[2.0 * 1.0 + 4.0 * 3.0]));
        assert_eq!(v["input"], tensor(&[2, 1, 1, 1], &[2.0, 4.0]));
    }

    #[test]
    fn batch_norm_backward_rejects_bad_args() {
        let input = tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let good = |p: &[f64]| tensor(&[2], p);
        let grad = tensor(&[1, 2, 1, 2], &[0.0; 4]);
        // rank / param / grad-shape mismatches and arity.
        assert!(
            call_fn(
                &BatchNormBackward,
                &[
                    tensor(&[4], &[0.0; 4]),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                    grad.clone()
                ]
            )
            .is_err()
        );
        assert!(
            call_fn(
                &BatchNormBackward,
                &[
                    input.clone(),
                    good(&[1.0; 2]),
                    good(&[0.0; 2]),
                    good(&[0.0; 2]),
                    good(&[1.0; 2]),
                    tensor(&[4], &[0.0; 4])
                ]
            )
            .is_err()
        );
        assert!(call_fn(&BatchNormBackward, &[input.clone()]).is_err());
    }
}
