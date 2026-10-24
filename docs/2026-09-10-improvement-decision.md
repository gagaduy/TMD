# RTM improvement decision

## Evidence inspected

- Published Pattern Recognition paper: `1-s2.0-S003132032400579X-main.pdf`.
- Released-checkpoint validation sweep: `.worktrees/iou30-combined/ASCFormer/work_dirs/iou30_combined/stage0_slide_val/metrics/threshold_sweep.json`.
- Boundary-FPN validation sweep: `.worktrees/iou30-combined/ASCFormer/work_dirs/boundary_fpn/official_real_val/metrics/threshold_sweep.json`.
- Decoder forward/predict, threshold metric and official evaluation implementation.

The baseline has IoU 22.5559%, precision 53.5309%, recall 28.0477% at threshold .25. At .05, recall rises to 36.4955% but precision falls to 27.8057% and IoU to 18.7390%. Boundary-FPN reaches 22.1919% at the lower boundary of its searched thresholds (.05); lower thresholds were not evaluated. Neither result establishes an architecture ceiling.

At fixed baseline true positives, removing every false positive would yield IoU equal to recall, 28.0477%. Thus false-positive suppression alone cannot reach 30% from that operating point. A successful change must recover missed positives, or jointly discriminate candidates from a lower-threshold operating point. Precision .60 and recall .40 would yield IoU .3158; these are illustrative joint targets, not predictions.

## Selected direction

Preserve the complete released ASC-Former and add a residual forensic-consistency adapter. The adapter compares local forensic features against several reference feature summaries from the same image/crop, and combines those differences with shallow image features and baseline logits. Its purpose is to recover missed regions while rejecting visually similar genuine content, rather than merely smoothing mask boundaries.

- Keep both pretrained streams and the original contrastive decoder.
- Reuse available transformed-domain features; do not import a third-party checkpoint or add OCR.
- Use multiple reference summaries rather than assuming all background is authentic. References from baseline probabilities are noisy and must be treated as such, never as ground truth. Cover/inpaint cases are an explicit risk.
- Predict a signed correction to logits with a zero-initialized output projection. Initial evaluation predictions must match the released model under identical inference settings.
- The adapter must operate densely, not only inside baseline-positive masks, so it can recover missed regions.
- Train with actual masks on official train.txt. Include baseline false positives and false negatives as hard examples from training data only, retaining ordinary examples and clean images. Do not mine validation masks into training.
- Use batch-independent normalization for new modules; frozen pretrained BatchNorm statistics must not accidentally freeze untrained statistics in new modules.

This is a selected research hypothesis, not a demonstrated improvement. Aggregate confusion counts do not identify whether missed pixels come from small text, entire missing regions or boundary errors. No per-image visual error audit was completed in this research pass; that limitation must remain explicit.

## Execution and decision criteria

First establish exact baseline equivalence with the added adapter. Then train on the official training split, initially freezing the released network and learning the adapter. Evaluate at 4,000 and 8,000 actual new optimizer updates; count these separately from diagnostic training. Reuse cached validation probabilities for threshold selection, extending the sweep if its optimum is at an endpoint. Record IoU, precision, recall, per-technique counts, and size-stratified errors. Any detailed error examples are diagnostic validation outputs, not training labels.

Compare against the released baseline with identical split, sliding-window settings, mask processing and threshold-selection procedure. The 22.56% validation baseline is not the repository's 19.71% test result. Validation above 30% permits selecting a model for final test; only held-out test above 30% satisfies the user's original test target. A later ablation with the same training budget is required to attribute improvement to the consistency component. Stop this implementation if official validation offers no useful gain; do not claim the whole architecture family is disproven.

## Primary research grounding

- RTM: https://doi.org/10.1016/j.patcog.2024.110828 — subtle manipulation traces, technique diversity, multimodal fusion and contrastive learning.
- ObjectFormer: https://arxiv.org/abs/2203.14681 — prototype-based and patch-level consistency modeling for manipulation localization.
- ADCD-Net: https://arxiv.org/abs/2507.16397 — document content/forensic separation and pristine prototypes; also identifies DCT block-alignment sensitivity.

These publications motivate the mechanism. Their reported gains are not evidence that this adaptation will exceed 30% on RTM.
