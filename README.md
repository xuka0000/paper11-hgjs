# paper11-hgjs

Research-code review snapshot for paper11. This is a work in progress, not a completed benchmark release or proof of the manuscript's claims.

## What is included

The current research implementation, training/evaluation entrypoints and statically resolved local Python imports. `review_manifest.json` binds every exported file to its source hash and exported hash. Only deployment paths/instance identities are redacted in the listed files; scientific production source has not been rewritten for publication.

Current evidence: **2560/3840 development; 0/5376 formal; full12 gate closed**. See `REVIEW_FINDINGS.md` before interpreting results.

## Entry points and environment

- `Agent/Publication/m2m_action_ranker.py`
- `Agent/Publication/m2m_adaptive_hypergraph.py`
- `Agent/Publication/m2m_action_dataset.py`
- `Agent/Run/m2m_reference_development_gpu.py`
- `Agent/Publication/m2m_formal_benchmark.py`
- `tests/test_m2m_action_ranker.py`
- `tests/test_m2m_adaptive_hypergraph.py`

Python3.11 was used for bounded local mechanism checks; actual author/native environments may require different locked versions. Set PYTHONPATH to the checkout root, `src`, `scripts`, and, for AIHCIR, `01_uav_language_grounding/benchmark/carlaair_v1`. `environment_observed.json` describes the review host, not a certified universal environment lock. External packages/source imports and dynamic calls are listed in `dependencies.json`; not every imported name is a pip distribution.

Run `python -B review_code_check.py` for source integrity and syntax. The implementation's module docstrings and CLI parsers define training/evaluation arguments. These commands require the corresponding frozen campaign, authorized data and checkpoints; do not invent substitute data or weaken admission checks to make them run.

## Reproduction boundary

Private manuscripts, production data, full model weights, cloud credentials and historical campaign directories are intentionally absent. Full scientific reproduction is **not yet verified**. The small Paper10 P0 example, where present, is synthetic and is not a paper result. Some tests require local author assets/campaign files; compile checks cannot certify their scientific behavior. Native simulation and external author sources must be obtained separately and matched to the registered versions.

## Third-party sources and rights

`source_links.json` records source URLs found in the implementation; module comments and adapter checks document attribution/version requirements. These links are not proof that a baseline has been fully reproduced. Do not represent locally adapted baselines as unmodified official implementations.

No new blanket license is granted by this snapshot. Existing third-party rights continue to apply; external author repositories/weights are not automatically relicensed or bundled. Request permission before redistribution where no applicable license is supplied.

## Review focus

Trace manuscript equations and pseudocode through the real training, checkpoint, inference, physical execution and independent evaluator. Distinguish synthetic diagnostics from measured results. Similar ablation means alone do not prove a bug or equivalence. Preserve negative evidence and frozen comparison budgets.
