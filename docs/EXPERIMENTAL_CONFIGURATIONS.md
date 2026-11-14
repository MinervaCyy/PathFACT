# Experimental Configurations

## 1. Scope and reproducibility boundary

This document records the historical configurations and evaluation relationships that can be supported by the available PathFACT release and approved experiment records. It distinguishes **training configurations** (model runs and optimization settings) from the Hugging Face dataset configuration identifiers (how released data files are loaded).

This is provenance documentation, not executable GPU training software. PathFACT's public framework is CPU-only. Researchers provide their own compatible environment for training and model inference. Fine-tuned model weights and optimizer/training-state files are not distributed. The available records do not independently establish complete reproduction of the historical runs.

## 2. Model and supervision datasets

The approved experiment records identify the model family as `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. An exact base-model revision and tokenizer revision are **Not verified** in the evidence available for this document.

The three SFT methods use matched G06 populations: 200 training examples and 44 loss-validation examples per method. The validation split is for teacher-forced loss, not independent task-solving evaluation.

| Method | Training supervision | Current public training file | Current public loss-validation file | HF dataset configuration |
|---|---|---|---|---|
| Vanilla-SFT | Correct final answer only | `vanilla_sft_train_200_prompt_completion_v1.jsonl` | `vanilla_sft_loss_val_44_prompt_completion_v1.jsonl` | `g06_vanilla_train_200`; `g06_vanilla_loss_val_44` |
| LIMO | Ordinary Teacher reasoning and answer | `limo_train_200_prompt_completion_v1.jsonl` | `limo_loss_val_44_prompt_completion_v1.jsonl` | `g06_gold_train_200`; `g06_gold_loss_val_44` |
| ContrastiveLIMO | Failure-conditioned corrective Teacher reasoning and answer | `contrastive_limo_train_200_prompt_completion_v1.jsonl` | `contrastive_limo_loss_val_44_prompt_completion_v1.jsonl` | `g06_contrastive_train_200`; `g06_contrastive_loss_val_44` |

These method-based filenames are the current v0.3 public paths. The record condition values remain `vanilla`, `gold`, and `contrastive`; `gold` denotes ordinary LIMO supervision. No dataset files are renamed or regenerated here. Inputs use prompt-completion records. Exact historical prompt templates, chat template, tokenizer behavior, sequence length, truncation, and tokenization implementation are **Not verified**.

## 3. Original training configuration

The canonical configuration below is reported for the completed Vanilla-SFT, LIMO, and ContrastiveLIMO experiments. It is the complete set of settings supported by the approved summary evidence; unspecified implementation details are explicitly left unverified.

| Parameter | Canonical value | Evidence status / note |
|---|---|---|
| Base model family | DeepSeek-R1-Distill-Qwen-7B | Reported in approved handoff; exact revision Not verified |
| Tokenizer and revision | Not verified | No run manifest available |
| Adaptation | LoRA | Reported in approved handoff |
| LoRA rank / alpha | 32 / 64 | Reported in approved handoff |
| LoRA dropout / target modules | Not verified | No primary run configuration available |
| Quantization / numerical precision | BF16; quantization Not verified | BF16 reported; quantization setting not established |
| Gradient checkpointing | Not verified | No primary run configuration available |
| Training/validation data | G06 Train-200 / Loss-Val-44 per method | Reported in approved handoff; public method files listed above |
| Loss | Completion-only supervised loss | Reported in approved handoff; exact masking and loss implementation Not verified |
| Maximum sequence length / truncation | Not verified | No primary run configuration available |
| Learning rate | 5e-5 | Reported in approved handoff |
| Optimizer and parameters | Not verified | Do not infer framework defaults |
| Weight decay | Not verified | No primary run configuration available |
| Scheduler | Linear with warmup | Reported in approved handoff; scheduler implementation details Not verified |
| Warmup | 10 steps | Reported in approved handoff |
| Per-device batch / gradient accumulation | Not verified | Effective batch is reported below; factorization is not available |
| Effective batch size | 64 | Reported in approved handoff |
| Gradient clipping | Not verified | No primary run configuration available |
| Epochs | 15 | Reported in approved handoff |
| Maximum steps | Not verified | No primary run configuration available |
| Random seed | 3407 | Reported in approved handoff |
| Logging / validation interval | Not verified | No primary run configuration available |
| Training framework and version | Not verified | No run environment manifest available |
| Completion status | Completed canonical SFT experiments | Approved evidence map identifies them as completed; exact scheduler/job records unavailable here |

The shared values above apply to the canonical three-method comparison. The supervision content differs by method as described in Section 2. The historical Vanilla-SFT run-to-training-file SHA256 mapping is not established: a known discrepancy affects the exact historical checkpoint-to-training-file hash attribution. The current public Vanilla files preserve the dataset release provenance but cannot resolve that historical mapping.

## 4. Training-recipe sensitivity

The available approved records describe these recipe differences:

| Recipe | Methods / status | Learning rate | Effective batch | Epochs | Scheduler / warmup | Seed | Evidence boundary |
|---|---|---:|---:|---:|---|---:|---|
| Original | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 5e-5 | 64 | 15 | Linear; 10 warmup steps | 3407 | Summary-level values; several optimizer and implementation details remain Not verified |
| Alternative 1 | LIMO and ContrastiveLIMO completed; no complete three-method downstream comparison | 1e-3 | 32 | 5 | Linear; 2 warmup steps | Not verified | Reported as validation-dynamics runs; do not treat as a completed three-way accuracy comparison |
| TrueWarm0 | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 1e-4 | 64 | 15 | Cosine; 0 warmup steps | 3407 | Summary-level values; optimizer and implementation details remain Not verified |
| TrueWarm0 Vanilla early stop | Separate Vanilla-SFT run stopped after 5 epochs | 1e-4 | 64 | 5 completed | Original 15-epoch cosine scheduler horizon retained; 0 warmup steps | 3407 | A distinct early-stopped run, not selection of epoch 5 from a completed 15-epoch run |

The Original-to-TrueWarm0 comparison changes learning rate and scheduler/warmup configuration together, so it does not isolate a single causal parameter. Alternative 1 is not evidence for a completed three-method result. The early-stopped Vanilla run is documented separately from the 15-epoch run and should not be described as an ordinary best-checkpoint selection.

The approved handoff reports task-only evaluation across G08 Balanced, G08 TrainDist, G09 Balanced, and G09 TrainDist for Original and TrueWarm0 E15 models, plus the separate TrueWarm0 Vanilla E5 run. These evaluation labels and result aggregates do not identify exact checkpoint artifacts or portable run IDs. Exact per-run checkpoint IDs, validation-best epochs, and selection criteria are **Not verified** here.

## 5. Run-to-evaluation traceability

The evidence supplied for this audit does not include the completed evidence register's run manifest, checkpoint metadata, frozen output manifest, or portable run identifiers. Therefore, the following rows map confirmed experiment conditions to their recorded evaluation views without inventing a run ID or checkpoint epoch.

| Run ID | Method | Recipe | Evaluated epoch | Evaluation views | Evidence / traceability |
|---|---|---|---:|---|---|
| Not verified | Vanilla-SFT | Original | 15 (reported) | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Approved handoff and evidence map identify canonical E15 condition; checkpoint ID and file-to-data hash mapping Not verified |
| Not verified | LIMO | Original | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Approved handoff and evidence map identify canonical E15 condition; checkpoint ID Not verified |
| Not verified | ContrastiveLIMO | Original | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Approved handoff and evidence map identify canonical E15 condition; checkpoint ID Not verified |
| Not verified | Vanilla-SFT | TrueWarm0 | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Approved handoff identifies TrueWarm0 E15; exact checkpoint metadata Not verified |
| Not verified | LIMO | TrueWarm0 | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Approved handoff identifies TrueWarm0 E15; exact checkpoint metadata Not verified |
| Not verified | ContrastiveLIMO | TrueWarm0 | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Approved handoff identifies TrueWarm0 E15; exact checkpoint metadata Not verified |
| Not verified | Vanilla-SFT | TrueWarm0 early stop | 5 completed | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Separate five-epoch early-stopped run; not a selected E5 checkpoint from a completed E15 run |
| Not verified | LIMO | Alternative 1 | 5 (reported) | Validation dynamics; downstream evaluation mapping Not verified | Approved handoff states no complete three-way downstream comparison |
| Not verified | ContrastiveLIMO | Alternative 1 | 5 (reported) | Validation dynamics; downstream evaluation mapping Not verified | Approved handoff states no complete three-way downstream comparison |

The Original/Stay/Switch evaluation contains eight conditions: shared Base; the three Original E15 methods; the three TrueWarm0 E15 methods; and TrueWarm0 Vanilla E5. Evaluation outputs and condition-level aggregates are described in the approved experiment map, but exact portable checkpoint identifiers and saved-result filenames are not present in the records available for this document. No private filesystem paths are reproduced.

## 6. Ranking-supervision configurations

The completed Single- and Dual-Ranking experiments directly support the approved manuscript scope. They augment **LIMO SFT**, not ContrastiveLIMO, with answer-level preference supervision. The available approved records identify G06 Train-200 and the 162 strict ranking cases; Single applies Switch ranking at weight 0.1, while Dual splits that weight across Original and Switch (0.05 each). The records describe separate runs from the base model.

Exact ranking optimizer settings, training framework version, checkpoint identifiers, evaluated epochs, and some loss-construction details are **Not verified** from the evidence available here. Ranking supervision remains distinct from the three SFT methods. Boundary-Local evidence is limited to candidate preference scoring; no completed Boundary-Local task-only VSP run is claimed.

## 7. Provenance and limitations

- Reported values in the tables are supported by the PathFACT ESWA handoff v1.2 and Contribution–Dataset–Experiment–Evidence Map v1.1. Original scripts, complete execution logs, the completed evidence register, and checkpoint manifests were not available for this documentation audit, so summary-level values are identified as such.
- Not verified values include exact base-model/tokenizer revisions, full prompt/chat serialization, tokenization and truncation, optimizer settings, weight decay, per-device batch and accumulation, clipping, maximum steps, intervals, framework/version, LoRA dropout/targets, and exact run/checkpoint identifiers.
- Historical checkpoint-to-training-file provenance is incomplete, particularly the exact Vanilla-SFT checkpoint-to-training-file SHA256 mapping.
- The approved records establish experiment-level method, recipe, and evaluation-view relationships, but not a complete portable mapping from each run to the exact evaluated checkpoint and saved output artifact.
- PathFACT does not publish full or adapter model weights, optimizer states, scheduler states, checkpoint archives, or other training-state artifacts. This absence limits direct reproduction of the reported model evaluations.
- GPU training and model inference were performed outside the public CPU-only framework and remain the researcher's responsibility in a compatible environment.
- Complete historical training reproduction has not been independently established. No framework defaults are substituted for missing records.

### Evidence records

- PathFACT ESWA Manuscript and Software Release Handoff v1.2 (project authority record).
- PathFACT ESWA Contribution–Dataset–Experiment–Evidence Map v1.1 (project authority record).
- Current public method-based dataset paths: [Dataset Map](../DATASET_MAP.md) and the [Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT).
