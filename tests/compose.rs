//! The composition seam, end to end: a runtime builds the `tensor`
//! `HandlModule` from this crate's `.hnl` interface + `OpRegistry`, plugs
//! it into a `ModuleRegistry`, and dispatches `tensor.*` qualified-name
//! calls through `ModuleLookup` — the same path a `call_host("tensor.matmul")`
//! from `.hnl` code takes.
//!
//! No language data or eval engine is involved: the module chain *is* the
//! seam this component exists to prove.

use std::sync::Arc;

use handl_host_tensor::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};
use handl_platform::{EvalContext, Module, ModuleLookup};
use handl_plugin_lib_handl::HandlModule;
use handl_plugin_module::ModuleRegistry;
use serde_json::{Value, json};

fn tensor(shape: &[usize], data: &[f64]) -> Value {
    json!({
        "__tensor__": {
            "shape": shape,
            "data": data,
            "dtype": "f64",
        }
    })
}

/// The module a runtime would build — `.hnl` interface + native impls.
fn tensor_module() -> HandlModule {
    HandlModule::new("tensor", TENSOR_HNL, &[]).with_ops(TENSOR_OPS, Arc::new(tensor_op_registry()))
}

#[test]
fn module_exposes_every_op_by_local_name() {
    let module = tensor_module();
    assert_eq!(module.name(), "tensor");
    for op in TENSOR_OPS {
        let local = op.rsplit('.').next().unwrap();
        assert!(module.function(local).is_some(), "missing {op}");
    }
}

#[test]
fn registry_lookup_dispatches_qualified_names() {
    let registry = ModuleRegistry::new().with(tensor_module());
    let matmul = registry
        .lookup("tensor.matmul")
        .expect("tensor.matmul should resolve through the module chain");
    let mut ctx = EvalContext::new();
    let out = matmul
        .call(
            &[
                tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]),
            ],
            &mut ctx,
        )
        .unwrap();
    assert_eq!(out, tensor(&[2, 2], &[19.0, 22.0, 43.0, 50.0]));
}

#[test]
fn registry_lookup_unknown_names_miss() {
    let registry = ModuleRegistry::new().with(tensor_module());
    assert!(registry.lookup("tensor.softmax").is_none());
    assert!(registry.lookup("other.zeros").is_none());
}

/// `tensor.conv2d` through the same module chain — the NCHW conv op
/// resolves and computes (1x1 kernel = per-pixel channel mix).
#[test]
fn registry_lookup_dispatches_conv2d() {
    let registry = ModuleRegistry::new().with(tensor_module());
    let conv2d = registry
        .lookup("tensor.conv2d")
        .expect("tensor.conv2d should resolve through the module chain");
    let mut ctx = EvalContext::new();
    let out = conv2d
        .call(
            &[
                tensor(&[1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]),
                tensor(&[2, 2, 1, 1], &[1.0, 2.0, -1.0, 0.0]),
            ],
            &mut ctx,
        )
        .unwrap();
    assert_eq!(out, tensor(&[1, 2, 1, 2], &[7.0, 10.0, -1.0, -2.0]));
}

/// A second host module composed alongside tensor — the point of the seam:
/// providers are independent rows in the consumer's registry.
#[test]
fn composes_with_other_modules() {
    use handl_platform::{EvalError, Function};

    struct Double;
    impl Function for Double {
        fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> Result<Value, EvalError> {
            let n = args[0]
                .as_f64()
                .ok_or(EvalError::TypeMismatch("num".into()))?;
            Ok(json!(n * 2.0))
        }
    }
    struct UtilModule(Double);
    impl Module for UtilModule {
        fn name(&self) -> &str {
            "util"
        }
        fn function(&self, name: &str) -> Option<&dyn Function> {
            (name == "double").then_some(&self.0 as &dyn Function)
        }
    }

    let registry = ModuleRegistry::new()
        .with(tensor_module())
        .with(UtilModule(Double));
    let mut ctx = EvalContext::new();
    assert_eq!(
        registry
            .lookup("util.double")
            .unwrap()
            .call(&[json!(21)], &mut ctx)
            .unwrap(),
        json!(42.0)
    );
    let shape = registry.lookup("tensor.shape").unwrap();
    assert_eq!(
        shape.call(&[tensor(&[2, 3], &[0.0; 6])], &mut ctx).unwrap(),
        json!([2, 3])
    );
}
