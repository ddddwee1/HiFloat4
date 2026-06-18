# Changelog

## 2026-06-18

### Fixed

- `mbsmxfp4`: changed the per-128 macro-factor target peak from `6.0` (E2M1_MAX)
  to `1.0`, in both the Torch reference (`quant_cy_npu/base/QFuncs/mbsmxfp4.py`,
  via a new `MACRO_FACTOR_TARGET` constant) and the AscendC kernel
  (`quant_cy_npu/base/cusrc/mbsmxfp4_quant_op.h`, the `six_s` duplicate in both
  the float and bf16 kernels). With `6.0`, an input whose per-128 amax lands near
  a power of two — int8 cast to float, amax ~ 128 = 2^7 — gets macro factor
  `M ~ 1.5`, which shifts every non-max inner group's e8m0 log-phase into the
  e2m1 clamp band and collapses SQNR (~ -7 dB vs plain MXFP4). `1.0` makes
  `M ~ 1.0` there so MBS degrades gracefully to plain MXFP4, and it also lifts
  the random-float case from ~ -1.9 dB to ~ +1.2 dB vs plain. The inner per-32
  e8m0/e2m1 path still uses `E2M1_MAX = 6.0`.

### Validated

- Rebuilt `npu_quant` (`build_npu_ops.sh`) and re-ran
  `quant_cy_npu/test_cases/mbsmxfp4.py` on real A2 hardware. NPU-vs-Torch max abs
  diff = `0.0` for float32 and bfloat16, on both random finite inputs and the
  all-zero case.

## 2026-06-15

Related paper:

- HiFloat4 Format for Language Model Inference
- [https://arxiv.org/abs/2602.11287](https://arxiv.org/abs/2602.11287)

### Added

- Migrated the EasyASC A2 vec-only `mbsmxfp4` kernel into
  `quant_cy_npu/base/cusrc/mbsmxfp4_quant_op.h`.
- Added the repo-local AscendC helper header
  `quant_cy_npu/base/cusrc/tensorutils_asc.h`.
- Added the `mbsmxfp4` Torch reference in
  `quant_cy_npu/base/QFuncs/mbsmxfp4.py`.
- Added `mbsmxfp4` wiring in `QType.py`, `QTensor.py`, and
  `quant_cy_npu/base/cusrc/npu_quant.cpp`.
- Added the NPU-vs-Torch smoke test
  `quant_cy_npu/test_cases/mbsmxfp4.py`.
- Added this migration record and the accompanying integration manual.

### Changed

- Kept the kernel migration source-faithful: the port stays close to the
  EasyASC-generated vec code and only adapts the target repo integration shell.
- Converted the work partition rule from the original `GetBlockNum() * 2`
  reasoning to `GetBlockNum()` for the pure vec target path.
- Set the validated `mbsmxfp4` launch blockDim to `40`.
- Registered the NPU kernel as BF16-only and left the float slot unset
  (`None`) until a dedicated float kernel exists.
- Renamed the local helper header from `mbsmxfp4_vec_support.h` to
  `tensorutils_asc.h`.

### Fixed

- Corrected the E8M0 reference path so the scale step follows floor semantics
  instead of rounding semantics.
- Added the required vec-pipe barriers for alias-sensitive vector instruction
  chains in the migrated kernel.
- Cleaned up temporary probe/debug scripts used during bring-up.

### Validated

- Verified the EasyASC path against the simulator and real A2 hardware.
- Verified the `hifx4_npu` port on real A2 hardware with the
  `quant_cy_npu/test_cases/mbsmxfp4.py` smoke test.
- Confirmed the current board test reports zero max diff for both random finite
  inputs and the all-zero case.

### Documentation

- Rewrote `easyasc_integration_manual.md` as a reusable migration guide for
  future EasyASC operator ports instead of a host-specific run log or a
  single-operator note.
- Removed versioned server access details from markdown and replaced remote
  examples with placeholders.
