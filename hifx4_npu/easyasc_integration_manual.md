# EasyASC Integration Manual for `hifx4_npu`

This document is a general guide for migrating EasyASC-generated kernels into
the `hifx4_npu` custom-op stack.

It is meant to guide future operator ports, not just one specific kernel. The
rules below were extracted from a validated A2 vec-only migration path and
should be treated as the default migration policy unless a later operator proves
that a different rule is required.

Related paper:

- HiFloat4 Format for Language Model Inference
- [https://arxiv.org/abs/2602.11287](https://arxiv.org/abs/2602.11287)

## 1. Goal

The goal of this manual is to provide a repeatable path for moving an EasyASC
kernel into this repo's NPU runtime.

Typical migration target:

- source: an EasyASC-generated kernel that already matches the intended math
- destination: `quant_cy_npu/base/cusrc`
- runtime wrapper: `quant_cy_npu/base/cusrc/npu_quant.cpp`
- higher-level integration: `QType.py`, `QTensor.py`, `QFuncs`, and a dedicated
  `test_cases` script

## 2. Recommended migration order

Use the following order for future ports:

1. Make the EasyASC source path correct first.
2. Validate the EasyASC result in simulator.
3. Validate the EasyASC result on real hardware.
4. Freeze the generated vec code as the migration source of truth.
5. Move the kernel into `hifx4_npu` with the smallest possible structural
   change.
6. Add the Torch reference path.
7. Add pybind, `QType`, `QTensor`, and test wiring.
8. Rebuild and validate on real hardware again inside `hifx4_npu`.

Do not start by redesigning the kernel inside the target repo. First preserve
behavior, then optimize later.

## 3. Source-of-truth policy

The safest default is to stay faithful to the generated EasyASC vec code.

Rules:

- Start from the verified EasyASC output, not from a handwritten rewrite of the
  algorithm.
- Keep instruction order, loop structure, local-tensor layout, and staging flow
  as close to the EasyASC source as possible.
- Do not split the original vec body into extra helper functions unless the
  target repo build system forces it.
- If the target repo needs a small amount of local support code, move that code
  into a support header instead of reshaping the main kernel body.

This repo currently uses:

- `quant_cy_npu/base/cusrc/tensorutils_asc.h`

for repo-local AscendC helper utilities.

## 4. Target repo touchpoints

For a typical quantization operator, the migration usually touches all of the
following layers.

### 4.1 Kernel body

Add a new kernel file under:

- `quant_cy_npu/base/cusrc/<op>_quant_op.h`

Default rule:

- pull the vec body over directly
- then adapt only the include list, namespace, `extern "C"` entry, and launcher

### 4.2 Support header

Only add a local support header when the kernel truly needs reusable helper
code.

Rules:

- keep the header repo-local
- keep it small
- avoid mixing math translation with build glue

### 4.3 Launcher and pybind

Wire the kernel in:

- `quant_cy_npu/base/cusrc/npu_quant.cpp`

Typical tasks:

- declare the `run_<op>_kernel...` launcher
- add the C++ wrapper that receives `at::Tensor`
- allocate workspace if needed
- choose an explicit `blockDim`
- export the pybind symbol

### 4.4 Torch reference

Add a reference implementation under:

- `quant_cy_npu/base/QFuncs/<op>.py`

This reference is used for correctness checks against the NPU kernel.

### 4.5 Quant type and dispatch

If the operator introduces a new quantization type, wire it through:

- `quant_cy_npu/base/QType.py`
- `quant_cy_npu/base/QTensor.py`

### 4.6 Test entry

Add a dedicated smoke test under:

- `quant_cy_npu/test_cases/<op>.py`

Do not rely on an unrelated test file to validate a new operator.

## 5. Kernel translation rules

### 5.1 Prefer vec-only translation unless the repo truly needs cube

For this repo, the default migration style is:

- keep only the vec side
- discard cube-side code unless the operator genuinely depends on it
- do not carry over mixed-environment boilerplate such as `AIC_V_1_2` setup
  when the target path is pure vec

### 5.2 Keep the vec body direct

Do not force the EasyASC output into a new class abstraction just because other
operators use a different style.

Preferred rule:

- bring over the vec body directly
- adapt the outer `extern "C"` interface and launcher only

This keeps debugging much easier when comparing target code with the generated
EasyASC source.

### 5.3 Preserve native AscendC operators

When the EasyASC-generated vec path already uses native AscendC interfaces,
preserve them.

Common examples:

- `WholeReduceMax`
- `Brcb`
- `CompareScalar`
- `Select`

Do not replace these with handwritten alternatives unless there is a verified
target-repo incompatibility.

### 5.4 Adjust block partitioning for pure vec mode

One of the most important migration differences is block-count interpretation in
pure vec mode.

Rule:

- if the original EasyASC-side reasoning used `GetBlockNum() * 2`
- and the target repo runs the kernel as pure vec mode
- then the migrated partition logic should become `GetBlockNum()`

Why:

- in this repo's pure vec path, `GetBlockNum()` already reflects the doubled
  vec-side view
- keeping the extra `* 2` would over-shard the work

When in doubt, validate the work split on real hardware before optimizing.

### 5.5 Be conservative about synchronization

Real-hardware mismatches can appear even when the translated code looks
mathematically identical.

Pay extra attention to chains that combine:

- reinterpret-style aliasing across dtypes
- bitwise exponent extraction
- `Brcb`
- `CompareScalar`
- `Select`
- immediate reuse by later vector consumers

If hardware output is wrong while the algorithm still looks correct, inspect
missing `PipeBarrier<PIPE_V>()` before rewriting the math.

### 5.6 Keep launch and workspace explicit

Do not hide launch assumptions.

For each migrated operator, make the following explicit in the C++ wrapper:

- launch blockDim
- whether the path is BF16-only or broader
- workspace size, if any

These values should be easy to inspect and easy to change per operator.

## 6. Dtype support policy

Do not pretend an operator supports dtypes that were never validated.

If only the BF16 kernel exists:

- wire only the BF16 NPU slot
- keep unsupported dtype slots as `None`

In `QTensor.py`, also keep the current fallback rule:

- if the requested dtype slot is `None`
- and the float slot is also `None`
- return `None` instead of silently forcing a fake fallback

This is important for incremental migration: it lets the repo support BF16
first, then add float support later as a separate kernel.

## 7. Validation workflow

### 7.1 Build locally

From the repo root:

```bash
cd quant_cy_npu/base/cusrc
python setup.py build_ext --inplace
```

### 7.2 Run the operator smoke test

From the repo root:

```bash
PYTHONPATH=. python quant_cy_npu/test_cases/<op>.py
```

Expected result:

- the script completes
- the reference path and kernel path agree on the tested inputs

### 7.3 Validate on real hardware

After local build success, re-run the same operator test on real hardware.

Machine-specific access details must stay out of versioned markdown. Use a local
ignored note for real usernames, real hosts, and real ports.

Versioned docs should use placeholders only:

```bash
rsync -avz -e 'ssh -p <remote_port>' \
    --exclude '.git' --exclude '__pycache__' --exclude 'build' \
    --exclude '*.so' --exclude '*.pyc' \
    ./ <remote_user>@<remote_host>:<remote_repo_root>/

ssh -p <remote_port> <remote_user>@<remote_host> 'bash -lc "
    cd <remote_repo_root> &&
    bash build_npu_ops.sh &&
    PYTHONPATH=. python quant_cy_npu/test_cases/<op>.py
"'
```

### 7.4 Validate special values separately when needed

Ordinary finite random tests are not enough for every operator.

When the quantization format can be affected by special values, add targeted
checks for:

- zero
- `nan`
- `+inf`
- `-inf`

Keep this as a separate validation step even if the base smoke test already
passes.

## 8. Practical migration checklist

Before calling a migration complete, verify all of the following:

- the EasyASC source path is already correct
- the migrated kernel still matches the EasyASC vec structure closely
- `npu_quant.cpp` exports the new operator
- `QType.py` recognizes the quant-type string, if a new type was introduced
- `QTensor.py` can route the operator correctly
- unsupported dtype slots remain explicit `None`
- the Torch reference exists
- the dedicated test case exists
- local build works
- board-level validation works
- versioned markdown contains no real host access information

## 9. Privacy rule for versioned docs

Do not store the following in versioned markdown under this repo:

- real usernames
- real hostnames
- real IP addresses
- real SSH ports
- machine-local absolute access notes

Keep those details in a local ignored note outside versioned docs.
