//! `tensor.derive` — the registered-adjoint table (not autodiff).
//!
//! [`DERIV_PAIRS`] maps each forward op to the qualified name of its
//! adjoint; [`Derive`] is the `Function` that answers the lookup.

use handl_platform::{EvalContext, EvalError, Function};
use serde_json::{Value, json};

use super::codec::type_mismatch;

/// Forward op name → registered adjoint op name, the table
/// `tensor.derive` reads. A pair is only listed when the adjoint impl
/// exists in [`tensor_op_registry`], so `derive`'s result always
/// resolves through `ModuleLookup::lookup` on the same module — a
/// `.hnl` optimization loop can call the returned name sight unseen.
pub const DERIV_PAIRS: &[(&str, &str)] = &[
    ("tensor.add", "tensor.add_backward"),
    ("tensor.mul", "tensor.mul_backward"),
    ("tensor.sub", "tensor.sub_backward"),
    ("tensor.div", "tensor.div_backward"),
    ("tensor.matmul", "tensor.matmul_backward"),
    ("tensor.conv2d", "tensor.conv2d_backward"),
    ("tensor.max_pool", "tensor.max_pool_backward"),
    ("tensor.reshape", "tensor.reshape_backward"),
    ("tensor.transpose", "tensor.transpose_backward"),
    ("tensor.concat", "tensor.concat_backward"),
    ("tensor.batch_norm", "tensor.batch_norm_backward"),
];

/// The Rust handl for `op "tensor.derive" : String -> Value`.
///
/// Returns the qualified op name of the adjoint registered for `name`
/// in [`DERIV_PAIRS`], or `null` when no pair exists. `name` may be
/// qualified (`"tensor.batch_norm"`) or a bare local name
/// (`"batch_norm"` → `"tensor.batch_norm"`). The result resolves
/// through `ModuleLookup::lookup` on the same registry, so an
/// optimization-loop consumer can dispatch it directly.
///
/// This is a *registered-pairs* table, not autodiff: a `.hnl` function
/// name gets `null` — a host `Function` sees only evaluated argument
/// values, so deriving an adjoint for `.hnl` source would need a
/// source-level transform or a reverse-mode tape in the eval engine
/// (README follow-up).
#[derive(Debug, Clone, Copy)]
pub struct Derive;
impl Function for Derive {
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
        handl_lib_ops::shared::require_arg_count(args, 1)?;
        let name = args[0]
            .as_str()
            .ok_or_else(|| type_mismatch("tensor.derive: op name (string)", "non-string"))?;
        let qualified = if name.contains('.') {
            name.to_string()
        } else {
            format!("tensor.{name}")
        };
        let adjoint = DERIV_PAIRS
            .iter()
            .find(|(f, _)| *f == qualified)
            .map(|(_, b)| *b);
        Ok(adjoint.map_or(Value::Null, |b| json!(b)))
    }
}
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::testutil::call_fn;
    use serde_json::json;

    #[test]
    fn derive_resolves_registered_pairs() {
        // Qualified and bare-local names both hit the pair.
        assert_eq!(
            call_fn(&Derive, &[json!("tensor.batch_norm")]).unwrap(),
            json!("tensor.batch_norm_backward")
        );
        assert_eq!(
            call_fn(&Derive, &[json!("batch_norm")]).unwrap(),
            json!("tensor.batch_norm_backward")
        );
    }

    #[test]
    fn derive_returns_null_for_unpaired_and_rejects_non_string() {
        assert_eq!(
            call_fn(&Derive, &[json!("tensor.zeros")]).unwrap(),
            Value::Null
        );
        assert_eq!(
            call_fn(&Derive, &[json!("my_hnl_function")]).unwrap(),
            Value::Null
        );
        assert!(call_fn(&Derive, &[json!(42)]).is_err());
        assert!(call_fn(&Derive, &[]).is_err());
    }
}
