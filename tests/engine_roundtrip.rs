//! Engine round-trip: a `.hnl` program drives `tensor.*` host ops through
//! the real `lex → parse → eval` pipeline — not just the module chain.
//!
//! `tests/compose.rs` proves the `serde_json::Value` module seam
//! (`ModuleRegistry::lookup` → `Function::call`). This file proves the next
//! hop, which is where `TaggedValue::Tensor` lives: inside an eval engine
//! the `{"__tensor__": …}` wire payload a host `Function` returns decodes
//! to the tagged variant (`call_host` does `TaggedValue::from(value)`), is
//! carried as an opaque value through `.hnl` code, and re-encodes to the
//! same wire form when it flows back out to a host call.
//!
//! Three assertions pin the round-trip down:
//!
//! 1. `EvalEngine::eval_tagged` returns [`TaggedValue::Tensor`] with shape
//!    and data intact for `tensor.zeros`/`tensor.matmul` results — the
//!    decode direction, asserted directly on the tagged value.
//! 2. Field access on the result errors with `type_tag == "tensor"`: a
//!    value still carrying the plain-`Obj` wire shape would project the
//!    `__tensor__` payload instead of failing.
//! 3. `Runtime::run` (tree-walk and bytecode) computes the right numbers
//!    through `from_list → matmul → {shape, to_list}` — the encode
//!    direction plus real host math on the decoded value.

use std::rc::Rc;
use std::sync::Arc;

use handl_host_tensor::{TENSOR_HNL, TENSOR_OPS, tensor_op_registry};
use handl_language::{Ast, Language, Source, Value};
use handl_platform::ModuleLookup;
use handl_plugin_lib_handl::HandlModule;
use handl_plugin_module::ModuleRegistry;
use handl_runtime::Runtime;
use handl_value::TaggedValue;
use hnl_engine_exec::{EvalEngine, EvalRequest};
use hnl_engine_exec_bytecode::BytecodeEngine;
use hnl_engine_exec_tree_walk::TreeWalkEngine;
use hnl_engine_lex_pattern::DataLexerEngine;
use hnl_engine_parse_pratt::PrattEngine;
use serde_json::json;

/// The canonical language (data-driven kit assembly — the same `Language`
/// the shipped toolchain runs).
fn canonical_lang() -> Language {
    handl_kit_data_canonical::assemble()
        .expect("canonical kit assembles")
        .lang()
        .clone()
}

/// The `tensor` module the runtime's `call_host` dispatch resolves
/// `tensor.*` string callees against — same construction as `compose.rs`.
fn tensor_modules() -> Rc<dyn ModuleLookup> {
    Rc::new(ModuleRegistry::new().with(
        HandlModule::new("tensor", TENSOR_HNL, &[])
            .with_ops(TENSOR_OPS, Arc::new(tensor_op_registry())),
    ))
}

/// A runtime with the canonical lexer/parser and the given eval engine.
/// `Runtime::with_modules` only *records* the registry — the module chain
/// must be wired into the eval engine itself for `call_host` dispatch.
fn runtime_with(eval: Box<dyn EvalEngine>) -> Runtime {
    Runtime::new(canonical_lang())
        .with_lex_engine(Box::new(DataLexerEngine::new()))
        .with_parse_engine(PrattEngine::new())
        .with_eval_engine(eval)
        .with_modules(tensor_modules())
}

fn tree_walk_runtime() -> Runtime {
    runtime_with(Box::new(
        TreeWalkEngine::new().with_modules(tensor_modules()),
    ))
}

fn bytecode_runtime() -> Runtime {
    runtime_with(Box::new(BytecodeEngine::with_modules(tensor_modules())))
}

/// `lex → parse` a `.hnl` snippet under the canonical language.
fn parse(rt: &Runtime, src: &str) -> Ast {
    let toks = rt.lex(&Source::new(src)).expect("lex must succeed");
    rt.parse(&toks).expect("parse must succeed")
}

/// A record literal bound to the `tensor` op names — the same shape
/// `from "host-tensor" import tensor` binds (`src/index.hnl`'s `export
/// tensor := {...}`), without needing the `import` machinery.
const TENSOR_BINDINGS: &str = r#"tensor := {
    zeros: "tensor.zeros",
    ones: "tensor.ones",
    from_list: "tensor.from_list",
    to_list: "tensor.to_list",
    shape: "tensor.shape",
    matmul: "tensor.matmul",
    add: "tensor.add"
};"#;

/// `tensor.zeros([2, 3])` evaluated by the tree-walk engine returns a
/// [`TaggedValue::Tensor`] — the host's `{"__tensor__": …}` wire payload
/// decoded at the `call_host` boundary, with shape and data intact.
#[test]
fn tree_walk_zeros_returns_tagged_tensor() {
    let rt = tree_walk_runtime();
    let ast = parse(
        &rt,
        &format!("do {{ {TENSOR_BINDINGS} tensor.zeros([2, 3]) }}"),
    );
    let engine = TreeWalkEngine::new().with_modules(tensor_modules());
    let value = engine
        .eval_tagged(rt.lang(), &EvalRequest::new(ast))
        .expect("eval_tagged must succeed");
    match value {
        TaggedValue::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 3]);
            assert_eq!(t.data.as_slice(), &[0.0; 6]);
        }
        other => panic!("expected TaggedValue::Tensor, got {other:?}"),
    }
}

/// `tensor.matmul` consumes two decoded tensors and returns a third —
/// decode → re-encode to the host (which validates shape and computes) →
/// decode again, all inside one `.hnl` evaluation.
#[test]
fn tree_walk_matmul_returns_tagged_tensor() {
    let rt = tree_walk_runtime();
    let ast = parse(
        &rt,
        &format!(
            "do {{ {TENSOR_BINDINGS} \
             a := tensor.from_list([[1, 2], [3, 4]]); \
             b := tensor.from_list([[5, 6], [7, 8]]); \
             tensor.matmul(a, b) }}"
        ),
    );
    let engine = TreeWalkEngine::new().with_modules(tensor_modules());
    let value = engine
        .eval_tagged(rt.lang(), &EvalRequest::new(ast))
        .expect("eval_tagged must succeed");
    match value {
        TaggedValue::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.data.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
        }
        other => panic!("expected TaggedValue::Tensor, got {other:?}"),
    }
}

/// The tensor is opaque inside the engine: field access `t.__tensor__`
/// must not see the wire object — it errors, and the diagnostic names the
/// tagged variant (`type_tag(TaggedValue::Tensor) == "tensor"`). If the
/// host's wire payload stayed a plain `Obj`, this would instead return the
/// `{shape, data, dtype}` payload — so this failing is the proof of
/// decode.
#[test]
fn tree_walk_tensor_is_not_a_record() {
    let rt = tree_walk_runtime();
    let err = rt
        .run(&Source::new(&format!(
            "do {{ {TENSOR_BINDINGS} \
             t := tensor.zeros([2, 3]); \
             t.__tensor__ }}"
        )))
        .expect_err("field access on a Tensor must fail, not project the wire payload");
    assert!(
        err.to_string().contains("tensor"),
        "error should name the 'tensor' type tag: {err}"
    );
}

/// End-to-end at the `Value` boundary on both eval engines: decode,
/// host-side matmul, and projection through `tensor.shape`/`tensor.to_list`
/// produce the right wire values.
#[test]
fn tensor_pipeline_round_trips_on_tree_walk_and_bytecode() {
    let src = format!(
        "do {{ {TENSOR_BINDINGS} \
         a := tensor.from_list([[1, 2], [3, 4]]); \
         b := tensor.from_list([[5, 6], [7, 8]]); \
         m := tensor.matmul(a, b); \
         {{ shape: tensor.shape(m), data: tensor.to_list(m), doubled: tensor.to_list(tensor.add(m, m)) }} }}"
    );
    for (name, rt) in [
        ("tree-walk", tree_walk_runtime()),
        ("bytecode", bytecode_runtime()),
    ] {
        let value: Value = rt
            .run(&Source::new(&src))
            .unwrap_or_else(|e| panic!("{name} run failed: {e}"));
        assert_eq!(
            value,
            json!({
                "shape": [2, 2],
                "data": [[19.0, 22.0], [43.0, 50.0]],
                "doubled": [[38.0, 44.0], [86.0, 100.0]],
            }),
            "{name}: matmul/shape/to_list pipeline"
        );
    }
}

/// A bare `tensor.zeros` program result at the `Value` boundary is the
/// wire form — the tagged tensor's canonical re-encoding — on both
/// engines.
#[test]
fn tensor_value_reencodes_to_wire_form() {
    for (name, rt) in [
        ("tree-walk", tree_walk_runtime()),
        ("bytecode", bytecode_runtime()),
    ] {
        let value = rt
            .run(&Source::new(&format!(
                "do {{ {TENSOR_BINDINGS} tensor.ones([3]) }}"
            )))
            .unwrap_or_else(|e| panic!("{name} run failed: {e}"));
        assert_eq!(
            value,
            json!({"__tensor__": {"shape": [3], "data": [1.0, 1.0, 1.0], "dtype": "f64"}}),
            "{name}: TaggedValue::Tensor must re-encode to the wire form"
        );
    }
}
