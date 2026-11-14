# Experimental Configurations

## 1. Scope and reproducibility boundary

This document records historical configurations and evaluation relationships for completed PathFACT experiments. It distinguishes **training configurations** (model runs and optimization settings) from the Hugging Face dataset configuration identifiers (how released data files are loaded).

This is provenance documentation, not executable GPU training software. PathFACT's public framework is CPU-only. Researchers provide their own compatible environment for training and model inference. Fine-tuned model weights and optimizer/training-state files are not distributed. The available records do not independently establish complete reproduction of the historical runs.

## 2. Model and supervision datasets

The reported model family is `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. An exact base-model revision and tokenizer revision are **Not verified** in the public evidence available for this document.

The three SFT methods use matched G06 populations: 200 training examples and 44 loss-validation examples per method. The validation split is for teacher-forced loss, not independent task-solving evaluation.

| Method | Training supervision | Current public training file | Current public loss-validation file | HF dataset configuration |
|---|---|---|---|---|
| Vanilla-SFT | Correct final answer only | `vanilla_sft_train_200_prompt_completion_v1.jsonl` | `vanilla_sft_loss_val_44_prompt_completion_v1.jsonl` | `g06_vanilla_train_200`; `g06_vanilla_loss_val_44` |
| LIMO | Ordinary Teacher reasoning and answer | `limo_train_200_prompt_completion_v1.jsonl` | `limo_loss_val_44_prompt_completion_v1.jsonl` | `g06_gold_train_200`; `g06_gold_loss_val_44` |
| ContrastiveLIMO | Failure-conditioned corrective Teacher reasoning and answer | `contrastive_limo_train_200_prompt_completion_v1.jsonl` | `contrastive_limo_loss_val_44_prompt_completion_v1.jsonl` | `g06_contrastive_train_200`; `g06_contrastive_loss_val_44` |

These method-based filenames are the current v0.3 public paths. The record condition values remain `vanilla`, `gold`, and `contrastive`; `gold` denotes ordinary LIMO supervision. No dataset files are renamed or regenerated here. Inputs use prompt-completion records. Exact historical prompt templates, chat template, tokenizer behavior, sequence length, truncation, and tokenization implementation are **Not verified**.

## 3. Original training configuration

The canonical configuration below is reported for the completed Vanilla-SFT, LIMO, and ContrastiveLIMO experiments. Unspecified implementation details are explicitly left unverified.

| Parameter | Canonical value | Evidence status / note |
|---|---|---|
| Base model family | DeepSeek-R1-Distill-Qwen-7B | Reported historical value; exact revision Not verified |
| Tokenizer and revision | Not verified | No run manifest available |
| Adaptation | LoRA | Reported historical value |
| LoRA rank / alpha | 32 / 64 | Reported historical values |
| LoRA dropout / target modules | Not verified | No primary run configuration available |
| Quantization / numerical precision | BF16; quantization Not verified | BF16 reported; quantization setting not established |
| Gradient checkpointing | Not verified | No primary run configuration available |
| Training/validation data | G06 Train-200 / Loss-Val-44 per method | Reported historical value; public method files listed above |
| Loss | Completion-only supervised loss | Reported historical value; exact masking and loss implementation Not verified |
| Maximum sequence length / truncation | Not verified | No primary run configuration available |
| Learning rate | 5e-5 | Reported historical value |
| Optimizer and parameters | Not verified | Do not infer framework defaults |
| Weight decay | Not verified | No primary run configuration available |
| Scheduler | Linear with warmup | Reported historical value; scheduler implementation details Not verified |
| Warmup | 10 steps | Reported historical value |
| Per-device batch / gradient accumulation | Not verified | Effective batch is reported below; factorization is not available |
| Effective batch size | 64 | Reported historical value |
| Gradient clipping | Not verified | No primary run configuration available |
| Epochs | 15 | Reported historical value |
| Maximum steps | Not verified | No primary run configuration available |
| Random seed | 3407 | Reported historical value |
| Logging / validation interval | Not verified | No primary run configuration available |
| Training framework and version | Not verified | No run environment manifest available |
| Completion status | Completed canonical SFT experiments | Reported as completed; exact scheduler/job records are not part of the public release |

The shared values above apply to the canonical three-method comparison. Vanilla-SFT uses correct final answers only, LIMO uses ordinary Teacher reasoning and answers, and ContrastiveLIMO uses failure-conditioned corrective Teacher reasoning and answers. The historical Vanilla-SFT run-to-training-file SHA256 mapping is not established: a known discrepancy affects the exact historical checkpoint-to-training-file hash attribution. The current public Vanilla files preserve the dataset release provenance but cannot resolve that historical mapping.

## 4. Training-recipe sensitivity

The reported recipe differences are:

| Recipe | Methods / status | Learning rate | Effective batch | Epochs | Scheduler / warmup | Seed | Evidence boundary |
|---|---|---:|---:|---:|---|---:|---|
| Original | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 5e-5 | 64 | 15 | Linear; 10 warmup steps | 3407 | Summary-level values; several optimizer and implementation details remain Not verified |
| Alternative 1 | LIMO and ContrastiveLIMO completed; no complete three-method downstream comparison | 1e-3 | 32 | 5 | Linear; 2 warmup steps | Not verified | Reported as validation-dynamics runs; do not treat as a completed three-way accuracy comparison |
| TrueWarm0 | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 1e-4 | 64 | 15 | Cosine; 0 warmup steps | 3407 | Summary-level values; optimizer and implementation details remain Not verified |
| TrueWarm0 Vanilla early stop | Separate Vanilla-SFT run stopped after 5 epochs | 1e-4 | 64 | 5 completed | Original 15-epoch cosine scheduler horizon retained; 0 warmup steps | 3407 | A distinct early-stopped run, not selection of epoch 5 from a completed 15-epoch run |

The Original-to-TrueWarm0 comparison changes learning rate and scheduler/warmup configuration together, so it does not isolate a single causal parameter. Alternative 1 is not evidence for a completed three-method result. The early-stopped Vanilla run is documented separately from the 15-epoch run and should not be described as an ordinary best-checkpoint selection.

Historical experiment records report task-only evaluation across G08 Balanced, G08 TrainDist, G09 Balanced, and G09 TrainDist for Original and TrueWarm0 E15 models, plus the separate TrueWarm0 Vanilla E5 run. These evaluation labels and result aggregates do not identify exact checkpoint artifacts or portable run IDs. Exact per-run checkpoint IDs, validation-best epochs, and selection criteria are **Not verified** here.

## 5. Run-to-evaluation traceability

The public release does not include a run manifest, checkpoint metadata, frozen output manifest, or portable run identifiers. Therefore, the following rows map reported experiment conditions to their recorded evaluation views without inventing a run ID or checkpoint epoch.

| Run ID | Method | Recipe | Evaluated epoch | Evaluation views | Evidence / traceability |
|---|---|---|---:|---|---|
| Not verified | Vanilla-SFT | Original | 15 (reported) | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Canonical E15 condition is reported; checkpoint ID and file-to-data hash mapping Not verified |
| Not verified | LIMO | Original | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Canonical E15 condition is reported; checkpoint ID Not verified |
| Not verified | ContrastiveLIMO | Original | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Canonical E15 condition is reported; checkpoint ID Not verified |
| Not verified | Vanilla-SFT | TrueWarm0 | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | TrueWarm0 E15 condition is reported; exact checkpoint metadata Not verified |
| Not verified | LIMO | TrueWarm0 | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | TrueWarm0 E15 condition is reported; exact checkpoint metadata Not verified |
| Not verified | ContrastiveLIMO | TrueWarm0 | 15 (reported) | Same four task-only views; Original/Stay/Switch counterfactual evaluation | TrueWarm0 E15 condition is reported; exact checkpoint metadata Not verified |
| Not verified | Vanilla-SFT | TrueWarm0 early stop | 5 completed | Same four task-only views; Original/Stay/Switch counterfactual evaluation | Separate five-epoch early-stopped run; not a selected E5 checkpoint from a completed E15 run |
| Not verified | LIMO | Alternative 1 | 5 (reported) | Validation dynamics; downstream evaluation mapping Not verified | No complete three-way downstream comparison is reported |
| Not verified | ContrastiveLIMO | Alternative 1 | 5 (reported) | Validation dynamics; downstream evaluation mapping Not verified | No complete three-way downstream comparison is reported |

The Original/Stay/Switch evaluation contains eight conditions: shared Base; the three Original E15 methods; the three TrueWarm0 E15 methods; and TrueWarm0 Vanilla E5. Condition-level aggregates are reported, but exact portable checkpoint identifiers and saved-result filenames are not present in the public release. No private filesystem paths are reproduced.

## 6. Ranking-supervision configurations

The completed Single- and Dual-Ranking experiments support the publication-relevant analysis. They augment **LIMO SFT**, not ContrastiveLIMO, with answer-level preference supervision. The recorded configuration uses G06 Train-200 and the 162 strict ranking cases; Single applies Switch ranking at weight 0.1, while Dual splits that weight across Original and Switch (0.05 each). These are separate runs from the base model.

Exact ranking optimizer settings, training framework version, checkpoint identifiers, evaluated epochs, and some loss-construction details are **Not verified** from the evidence available here. Ranking supervision remains distinct from the three SFT methods. Boundary-Local evidence is limited to candidate preference scoring; no completed Boundary-Local task-only VSP run is claimed.

## 7. Historical Experimental Hardware and Software Environments

This section records historical execution environments separately for generation, SFT, and model evaluation. **Execution verified** means a completed run has matching runtime evidence (and, for hardware, an allocation or resource record); **Configured** means the value appears in a launch script or configuration but execution was not independently confirmed; **Not verified** means the available records do not establish the value. Internal paths, account names, and job identifiers are omitted. These are historical environments, not dependencies of the public CPU-only framework.

### 7.1 Student–Teacher generation environment

Completed Student generation for DeepSeek-R1-Distill-Qwen-7B is execution-verified on one AMD Instinct MI355X GPU per generation process, using vLLM `0.11.2.dev673+g839868462` with the ROCm backend. The recorded run used BF16, tensor parallelism 1, no quantization, maximum model length 32,768, temperature 0.6, top-p 0.95, and a 16,384-token completion limit. The exact model and tokenizer revisions were not pinned in the captured run configuration. Python 3.12 is indicated by the runtime package path; its patch version is **Not verified**. PyTorch, Transformers, and the exact ROCm runtime versions are **Not verified** for this generation run.

Completed production Teacher generation for DeepSeek-R1-Distill-Qwen-32B is execution-verified with vLLM `0.11.2.dev673+g839868462`, BF16, tensor parallelism 1, no quantization, maximum model length 32,768, temperature 0.6, top-p 0.95, and an 8,192-token completion limit. The production launch requested one AMD Instinct MI300X GPU. A later Teacher generation wave requested one AMD Instinct MI355X GPU. Thus the recorded Teacher-generation campaigns did not all use the same GPU model. The available records do not establish a complete mapping from every released Teacher output to a specific hardware wave. Python 3.12 is indicated in runtime paths, but the patch version, PyTorch and Transformers versions, and exact ROCm runtime version are **Not verified**.

### 7.2 SFT training environment

| Completed SFT configuration | Hardware evidence | Training software evidence | Status and scope |
|---|---|---|---|
| Original Vanilla-SFT, LIMO, ContrastiveLIMO | One AMD Instinct MI355X GPU is recorded for each method in the completed runtime-cost record. | The exact framework and package versions for these particular runs are **Not verified**. Historical training scripts configure Unsloth, but the available evidence does not tie that configuration to an execution manifest for each canonical run. | Hardware execution verified; framework/version not verified per run. |
| TrueWarm0 (all three methods), Vanilla-SFT early-stopping run, Alternative 1 LIMO and ContrastiveLIMO | **Not verified** per configuration. | **Not verified** per configuration. | Do not generalize the Original hardware or framework record to these runs. |
| Completed Replay10 SFT runs (supporting replay) | Launch configuration requests one AMD Instinct MI355X GPU; captured runtime metadata reports an available AMD GPU but does not identify its exact model. | Completed run metadata records Unsloth 2026.8.18, TRL 0.24.0, PyTorch 2.10.0+rocm7.1, Transformers 4.57.6, and Python 3.12.8. The launch script requests the ROCm 7.14 module; the exact host module loaded at runtime is not independently confirmed. | Execution and package versions verified for these supporting replay runs only. This is not evidence for the canonical Original or TrueWarm0 environments. |

For the completed Replay10 runs, the captured configuration records BF16 and 16-bit loading, without 4-bit or 8-bit model loading, and LoRA with rank 32, alpha 64, and dropout 0. The recorded adapter target list is the attention and MLP projection layers. The configuration uses Unsloth gradient checkpointing and an 8-bit AdamW optimizer. These details apply only to the Replay10 runs for which the environment metadata was captured. The records do not establish that MI300X and Unsloth were used together for SFT. In particular, the completed canonical three-method cost record identifies MI355X; the completed Replay10 run requesting MI355X is the run with captured Unsloth package versions.

The Original method cost record reports training durations of 191.9927 seconds for Vanilla-SFT, 768.1175 seconds for LIMO, and 2187.5204 seconds for ContrastiveLIMO, each on one GPU. These recorded durations are retained as historical measurements and should not be treated as general hardware benchmarks.

### 7.3 Model inference and evaluation environment

The completed G08/G09 evaluation launch records request one AMD Instinct MI355X GPU and use the ROCm-enabled vLLM environment. The G09 runtime log confirms vLLM `0.11.2.dev673+g839868462`; G08 uses the same recorded launch configuration, while its exact runtime package-version capture is **Not verified**. The verified generation settings are BF16, tensor parallelism 1, no quantization, maximum model length 32,768, temperature 0.6, top-p 0.95, and maximum generation length 16,384 tokens. The evaluation client configuration records concurrency 8. Python 3.12 is indicated by runtime package paths, but the patch version, PyTorch and Transformers versions, and exact ROCm runtime version are **Not verified**.

vLLM was used to generate model predictions; it was not the shortest-path correctness verifier. Evaluation proceeded from saved prediction records through answer extraction and then deterministic CPU-based path verification and offline evaluation. These are distinct stages, and the public PathFACT framework remains CPU-only.

### 7.4 Environment-to-experiment mapping

| Experiment group | Training or generation environment | Evaluation environment | Evidence status |
|---|---|---|---|
| Student 7B generation | One MI355X GPU; vLLM version and settings are listed in §7.1 of this guide. | Not applicable to generation itself. | Execution verified for the recorded generation run. |
| Teacher 32B generation | Production wave: one MI300X GPU. Later wave: one MI355X GPU. vLLM version and settings are listed in §7.1 of this guide. | Not applicable to generation itself. | Execution verified for recorded waves; mapping of all released outputs to a specific wave is incomplete. |
| Original Vanilla-SFT, LIMO, ContrastiveLIMO | One MI355X GPU per method; framework and package versions not verified for these runs. | G08/G09 evaluation launch records request one MI355X GPU and vLLM; see §7.3 of this guide. | Training hardware execution verified; exact run/checkpoint-to-environment and output mapping remains limited as described in §8 of this guide. |
| TrueWarm0 methods and Vanilla-SFT early stopping | **Not verified** per run. | The evaluation records document campaign-level G08/G09 settings, but a complete mapping from each evaluated checkpoint to a specific execution record is **Not verified**. | Do not assume the Original training environment applied. |
| Alternative 1 LIMO and ContrastiveLIMO | **Not verified** per run. | No completed three-method accuracy comparison is claimed. | Training environment not verified; downstream comparison incomplete. |
| Single- and Dual-Ranking supervision | **Not verified**. | Evaluation environment for each ranking checkpoint is **Not verified**. | Do not infer an environment from the SFT or replay runs. |
| Completed Replay10 SFT | One-GPU MI355X requested; captured environment and Unsloth versions are listed in §7.2 of this guide. | A checkpoint-specific inference environment mapping is **Not verified** here. | Supporting replay only; separate from canonical comparison. |

These environment records supplement the run and evaluation metadata in the preceding sections. They do not establish a complete portable mapping from every historical run to its exact training artifact, evaluated checkpoint, and saved prediction file.

### 7.5 Software versions and evidence confidence

| Stage | Hardware | Framework | Verified version information | Evidence status |
|---|---|---|---|---|
| Student generation | 1 × AMD Instinct MI355X | vLLM, ROCm backend | vLLM `0.11.2.dev673+g839868462`; Python 3.12 minor series only | Execution verified; Python patch, PyTorch, Transformers, and ROCm versions not verified. |
| Teacher generation | Production: 1 × AMD Instinct MI300X; later wave: 1 × MI355X | vLLM, ROCm backend | vLLM `0.11.2.dev673+g839868462`; Python 3.12 minor series only | Execution verified for recorded campaigns; complete output-to-wave mapping and runtime versions not verified. |
| Canonical SFT | 1 × AMD Instinct MI355X per reported Original method run | Framework/version not verified per run | No executed-run package snapshot available | Hardware execution verified; software not verified. |
| Replay10 SFT | One GPU; launcher requests MI355X | Unsloth with TRL | Python 3.12.8; PyTorch 2.10.0+rocm7.1; Transformers 4.57.6; TRL 0.24.0; Unsloth 2026.8.18 | Captured completed-run metadata; exact physical GPU model and host ROCm module not independently verified. |
| G08/G09 inference | 1 × AMD Instinct MI355X requested | vLLM, ROCm backend | G09 vLLM `0.11.2.dev673+g839868462`; Python 3.12 minor series only | G09 execution verified; G08 runtime version, Python patch, PyTorch, Transformers, and ROCm versions not verified. |

### 7.6 Limitations

The records support use of AMD MI300X for a completed Teacher-generation campaign, but do not support a claim that MI300X was used with Unsloth for the completed SFT experiments. MI355X is the hardware recorded for the canonical SFT cost comparison and for G08/G09 evaluation launches. Unsloth execution and package versions are captured for supporting Replay10 runs; the canonical method runs’ exact training framework and package versions remain **Not verified**. The precise ROCm runtime version for generation/evaluation, exact package versions for canonical SFT, and per-run mappings for several sensitivity and ranking configurations are incomplete. Hardware and software differences across experiments must not be interpreted as causes of accuracy differences without a controlled comparison. These historical records do not establish complete independent reproducibility.

## 8. Provenance and limitations

- Training values are reported historical settings; exact execution records and checkpoint manifests are not part of the public release. Accordingly, summary-level values are marked as reported and operational details without public verification remain **Not verified**.
- Not verified values include exact base-model/tokenizer revisions, full prompt/chat serialization, tokenization and truncation, optimizer settings, weight decay, per-device batch and accumulation, clipping, maximum steps, intervals, framework/version, LoRA dropout/targets, and exact run/checkpoint identifiers.
- Historical checkpoint-to-training-file provenance is incomplete, particularly the exact Vanilla-SFT checkpoint-to-training-file SHA256 mapping.
- Publicly available records establish experiment-level method, recipe, and evaluation-view relationships, but not a complete portable mapping from each run to the exact evaluated checkpoint and saved output artifact.
- PathFACT does not publish full or adapter model weights, optimizer states, scheduler states, checkpoint archives, or other training-state artifacts. This absence limits direct reproduction of the reported model evaluations.
- GPU training and model inference were performed outside the public CPU-only framework and remain the researcher's responsibility in a compatible environment.
- Complete historical training reproduction has not been independently established. No framework defaults are substituted for missing records.

### Evidence records

- Current public method-based dataset paths: [Dataset Map](../DATASET_MAP.md) and the [Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT).
