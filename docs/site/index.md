# host-tensor

Host-side `tensor.*` functions: a dense n-d `f64` array for `.hnl` runtimes.

## What it is

Runtimes compose this component to give `.hnl` code tensor operations: a
dense n-dimensional `f64` array (row-major) behind the `tensor.*` op surface.
`src/index.hnl` is the single source of truth for the surface — the Rust side
implements exactly what that file declares. Catopt is the first consumer.

## Modules
- `src/index.hnl` — the declared op surface — the contract the Rust implements
- `src/ops/` — the Rust implementations (forward and backward passes)


## Start here

- **Reference** — [public API](/host-tensor/reference/)
- **How-to** — [task recipes](/host-tensor/how-to/)
- **Explanation** — [design and rationale](/host-tensor/explanation/)

No tutorials: this repo is a building block used from Rust. End-to-end
walkthroughs live with the product that composes it — see the
[Handl platform](/handl/).
