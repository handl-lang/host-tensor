# Reference

## Public API

| item | what it is |
|---|---|
| `tensor.zeros`, `tensor.ones`, `tensor.from_list` | construction |
| `tensor.to_list`, `tensor.shape` | inspection |
| `tensor.add`, `tensor.mul`, `tensor.sub`, `tensor.div` | elementwise arithmetic (numpy broadcasting) |
| `tensor.matmul`, `tensor.linear` | matrix multiply (rank-2/3, batch broadcast); `F.linear` |
| `tensor.reshape`, `tensor.unsqueeze`, `tensor.select`, `tensor.transpose`, `tensor.concat` | shape manipulation |
| `tensor.conv2d`, `tensor.max_pool`, `tensor.batch_norm` | the layer ops |
| `*_backward` variants | gradients for every differentiable op |
| `tensor.derive` | the registered adjoint op name for a primal (`null` when none) |

## Crate

| crate | role |
|---|---|
| `handl-host-tensor` | host functions (`tensor.*`) |

## Build and test

```bash
cargo build     # sibling [patch] tables resolve deps from ../<repo>
cargo test      # cargo test
```

Dependencies are git deps on `branch = "main"`; the `[patch]` tables
point at sibling checkouts for local development — CI strips them and
resolves the declared git selectors.
