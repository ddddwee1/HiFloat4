# Changelog

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
