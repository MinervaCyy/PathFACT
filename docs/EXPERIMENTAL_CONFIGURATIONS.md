# Experimental Configurations

## 1. Scope and reproducibility boundary

This document records historical configurations and evaluation relationships for completed PathFACT experiments. It distinguishes **training configurations** (model runs and optimization settings) from the Hugging Face dataset configuration identifiers (how released data files are loaded).

This is provenance documentation, not executable GPU training software. PathFACT's public framework is CPU-only. Researchers provide their own compatible environment for training and model inference. Fine-tuned model weights and optimizer/training-state files are not distributed. The available records do not independently establish complete reproduction of the historical runs.

## 2. Model and supervision datasets

The reported SFT base-model family is `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. The retained canonical SFT records do not identify the exact model or tokenizer revision used to initialize those runs. A resolved model revision is available for the separate 7B Student generation run; it does not establish the SFT initialization revision.

The three SFT methods use matched G06 populations: 200 training examples and 44 loss-validation examples per method. The validation split is for teacher-forced loss, not independent task-solving evaluation.

| Method | Training supervision | Current public training file | Current public loss-validation file | HF dataset configuration |
|---|---|---|---|---|
| Vanilla-SFT | Correct final answer only | `vanilla_sft_train_200_prompt_completion_v1.jsonl` | `vanilla_sft_loss_val_44_prompt_completion_v1.jsonl` | `g06_vanilla_train_200`; `g06_vanilla_loss_val_44` |
| LIMO | Ordinary Teacher reasoning and answer | `limo_train_200_prompt_completion_v1.jsonl` | `limo_loss_val_44_prompt_completion_v1.jsonl` | `g06_gold_train_200`; `g06_gold_loss_val_44` |
| ContrastiveLIMO | Failure-conditioned corrective Teacher reasoning and answer | `contrastive_limo_train_200_prompt_completion_v1.jsonl` | `contrastive_limo_loss_val_44_prompt_completion_v1.jsonl` | `g06_contrastive_train_200`; `g06_contrastive_loss_val_44` |

These method-based filenames are the current v0.3 public paths. The record condition values remain `vanilla`, `gold`, and `contrastive`; `gold` denotes ordinary LIMO supervision. Each released file contains explicit `prompt` and `completion` text. Canonical SFT run records do not preserve the tokenizer revision, the exact trainer-side serialization/tokenization procedure, or the sequence-length and truncation settings.

## 3. Original training configuration

The canonical configuration below reports settings available in the completed Vanilla-SFT, LIMO, and ContrastiveLIMO experiment summaries. Execution-level details that are absent from run-matched records are listed after the table.

| Parameter | Canonical reported value |
|---|---|
| Base model family | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Adaptation | LoRA; rank 32, alpha 64 |
| Numerical precision | BF16 (reported) |
| Training and loss-validation data | G06 Train-200 and Loss-Val-44 for each method |
| Loss | Completion-only supervised loss (reported) |
| Learning rate | 5e-5 |
| Scheduler and warmup | Linear scheduler (reported); 10 warmup steps |
| Effective batch size | 64 |
| Epochs | 15 |
| Random seed | 3407 |
| Completion | Vanilla-SFT, LIMO, and ContrastiveLIMO completed; the recorded runtime-cost report assigns one MI355X GPU to each method |

The shared values apply to the canonical three-method comparison. Vanilla-SFT uses correct final answers only, LIMO uses ordinary Teacher reasoning and answers, and ContrastiveLIMO uses failure-conditioned corrective Teacher reasoning and answers. The completed cost record reports 191.9927 seconds for Vanilla-SFT, 768.1175 seconds for LIMO, and 2187.5204 seconds for ContrastiveLIMO. These are historical measurements, not general hardware benchmarks.

The retained canonical G06 records contain a recipe summary and the runtime-cost report, but no run-matched trainer configuration, training stdout, environment snapshot, or trainer-state file. Consequently, the canonical runs' exact model/tokenizer revisions, tokenizer and truncation implementation, LoRA dropout and target modules, quantization flags, gradient checkpointing, masking implementation, maximum sequence length, optimizer and its parameters, weight decay, scheduler implementation, per-device batch/accumulation factorization, clipping, maximum steps, logging/validation interval, and training-framework version cannot be recovered from the retained execution evidence. No run-matched execution record establishes whether Unsloth or another training framework was used for the canonical three-method experiments.

## 4. Training-recipe sensitivity

The reported recipe differences are:

| Recipe | Methods / status | Learning rate | Effective batch | Epochs | Scheduler / warmup | Seed | Evidence boundary |
|---|---|---:|---:|---:|---|---:|---|
| Original | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 5e-5 | 64 | 15 | Linear; 10 warmup steps | 3407 | Recipe summary retained; optimizer and implementation details are absent from the canonical run records. |
| Alternative 1 | LIMO and ContrastiveLIMO completed; no complete three-method downstream comparison | 1e-3 | 32 | 5 | Linear; 2 warmup steps | Not present in the retained recipe record | Reported as validation-dynamics runs; do not treat as a completed three-way accuracy comparison |
| TrueWarm0 | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 1e-4 | 64 | 15 | Cosine; 0 warmup steps | 3407 | Recipe summary retained; optimizer and implementation details are absent from per-run records. |
| TrueWarm0 Vanilla early stop | Separate Vanilla-SFT run stopped after 5 epochs | 1e-4 | 64 | 5 completed | Original 15-epoch cosine scheduler horizon retained; 0 warmup steps | 3407 | A distinct early-stopped run, not selection of epoch 5 from a completed 15-epoch run |

The Original-to-TrueWarm0 comparison changes learning rate and scheduler/warmup configuration together, so it does not isolate a single causal parameter. Alternative 1 is not evidence for a completed three-method result. The early-stopped Vanilla run is documented separately from the 15-epoch run and should not be described as an ordinary best-checkpoint selection.

Historical experiment records report task-only evaluation across G08 Balanced, G08 TrainDist, G09 Balanced, and G09 TrainDist for Original and TrueWarm0 E15 models, plus the separate TrueWarm0 Vanilla E5 run. These evaluation labels and result aggregates do not identify exact checkpoint artifacts or portable run IDs. The retained records contain no canonical best-validation epoch or checkpoint-selection rule.

## 5. Run-to-evaluation traceability

No portable run IDs or checkpoint manifests are retained for these canonical runs. The following table maps the reported method/recipe conditions to the evaluation views and epochs recorded in the experiment summaries; it does not identify exact checkpoint files or a best-validation-loss selection rule.

| Methods | Recipe | Evaluated epoch | Evaluation views | Evidence boundary |
|---|---|---:|---|---|
| Vanilla-SFT, LIMO, ContrastiveLIMO | Original | 15 (reported) | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Condition and epoch are reported; exact checkpoint IDs and checkpoint-selection metadata are absent. |
| Vanilla-SFT, LIMO, ContrastiveLIMO | TrueWarm0 | 15 (reported) | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Condition and epoch are reported; exact checkpoint IDs and checkpoint-selection metadata are absent. |
| Vanilla-SFT | TrueWarm0 early stop | 5 completed | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Separate five-epoch early-stopped run, not an E5 checkpoint selected from a completed E15 run. |
| LIMO, ContrastiveLIMO | Alternative 1 | 5 (reported) | Validation dynamics only; no completed three-method accuracy comparison | Downstream evaluation/checkpoint mapping is not retained. |

The Original/Stay/Switch evaluation contains eight conditions: shared Base; the three Original E15 methods; the three TrueWarm0 E15 methods; and TrueWarm0 Vanilla E5. Condition-level aggregates are reported, but exact portable checkpoint identifiers and saved-result filenames are not present in the public release. No private filesystem paths are reproduced.

## 6. Ranking-supervision configurations

The completed Single- and Dual-Ranking experiments augment **LIMO SFT**, not ContrastiveLIMO, with answer-level preference supervision. The recorded configuration uses G06 Train-200 and the 162 strict ranking cases; Single applies Switch ranking at weight 0.1, while Dual splits that weight across Original and Switch (0.05 each). These are separate runs from the base model.

The retained ranking records establish those data and weight settings but contain no matching training command, runtime log, package snapshot, checkpoint registry, or evaluation-checkpoint mapping. Thus the optimizer, scheduler, batch/epoch/seed settings, exact ranking-loss implementation, framework/version, training GPU, evaluated epoch, and checkpoint-selection rule cannot be recovered. Ranking remains distinct from the three SFT methods. Boundary-Local evidence is limited to candidate preference scoring; no completed Boundary-Local task-only VSP run is claimed.

## 7. Historical Experimental Hardware and Software Environments

The tables below distinguish completed execution records from launch-only configuration. A resource request plus a matching completed runtime log supports the recorded GPU allocation; a script value without a matching run record is labelled as configured. Internal paths, accounts, and job identifiers are omitted. These historical environments are separate from the public CPU-only framework.

### 7.1 Student–Teacher generation environments

| Environment ID / campaign | Model and revision | Hardware | Runtime versions and evidence |
|---|---|---|---|
| GEN-S — Student production, 2,200 cases | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`; resolved model revision `916b56a44061fd5cd7d6a8fb632557ed4f724f60`. Tokenizer class `Qwen2TokenizerFast`; its snapshot revision is absent from the run manifest. | One AMD Instinct MI300X GPU. The completed run configuration, runtime manifest, and ROCm vLLM startup log agree on this production run. | Python 3.12.12; PyTorch `2.9.0a0+git1c57644`; PyTorch HIP build `7.0.51831-a3e329ad8`; Transformers 4.57.6; vLLM `0.11.2.dev673+g839868462`; ROCm backend. |
| GEN-T300 — Teacher production, Gold and Contrastive batches of 552 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`; engine log records `revision=None` and `tokenizer_revision=None`, so no immutable model/tokenizer revision was pinned. | One AMD Instinct MI300X GPU per batch. Launch requests and completed vLLM runtime logs support the allocation. | vLLM `0.11.2.dev673+g839868462`; ROCm backend. The matched run records do not capture Python, PyTorch, Transformers, or exact host ROCm runtime versions for these MI300X jobs. |
| GEN-T355 — Later Teacher wave, Gold and Contrastive shards | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`; the engine logs do not pin model or tokenizer revisions. | Four completed one-GPU shard jobs on AMD Instinct MI355X (two Gold shards and two Contrastive shards). | Python 3.12.12; PyTorch `2.9.0a0+git1c57644`; PyTorch HIP build `7.0.51831-a3e329ad8`; Transformers 4.57.6; vLLM `0.11.2.dev673+g839868462`; ROCm backend. Completed run configuration ties these package versions to the same frozen container image as the Student runtime manifest. |

| Campaign | vLLM engine startup settings | Per-request generation settings |
|---|---|---|
| GEN-S | Tensor parallel 1; BF16; no quantization; maximum model length 32,768; GPU-memory utilization 0.85; reasoning parser `deepseek_r1`; engine seed 0. | Temperature 0.6; top-p 0.95; maximum completion 16,384 tokens; client concurrency 1; request timeout 21,600 seconds. Each case receives a deterministic seed derived from its case identifier. |
| GEN-T300 | Tensor parallel 1; BF16; no quantization; maximum model length 32,768; GPU-memory utilization 0.85; reasoning parser `deepseek_r1`; engine seed 0. | Temperature 0.6; top-p 0.95; maximum completion 8,192 tokens; client concurrency 1; request timeout 21,600 seconds; request seed derived from paired case record, with CLI fallback 0. Structured output uses a strict JSON schema. |
| GEN-T355 | Tensor parallel 1; BF16; no quantization; maximum model length 32,768; GPU-memory utilization 0.85; reasoning parser `deepseek_r1`; engine seed 0. | Temperature 0.6; top-p 0.95; maximum completion 8,192 tokens for Gold and 14,336 for Contrastive; client concurrency 8; request timeout 21,600 seconds; request seed from paired case record, with fallback 0. |

The captured launch commands and request records do not set top-k, `max_num_seqs`, or `max_num_batched_tokens`; no values are assigned to those fields here. A separate Student MI355X submission script is retained, but no matching completed result configuration or runtime log was found, so it is not reported as an executed Student generation wave.

### 7.2 SFT training environments

| Completed experiment | Hardware | Training framework and versions | Evidence boundary |
|---|---|---|---|
| Original Vanilla-SFT, LIMO, ContrastiveLIMO | One AMD Instinct MI355X GPU per method in the completed runtime-cost record. | The completed canonical runs have no matching environment manifest or training log identifying Unsloth or another framework/version. | MI355X and runtimes are recorded; framework/version cannot be assigned to these runs. |
| TrueWarm0 methods; TrueWarm0 Vanilla early stop; Alternative 1 LIMO and ContrastiveLIMO | No per-run GPU allocation record survives for these recipes. | No per-run framework or package snapshot survives. | The Original MI355X record is not generalized to these recipes. Alternative 1 has no completed three-method accuracy comparison. |
| Completed Single- and Dual-Ranking | No per-run GPU allocation record survives. | No run-matched training framework/version record survives. | The retained records describe supervision weights and examples, not the executed training environment. |
The Stage C runtime-cost record reports 191.9927 seconds for Vanilla-SFT, 768.1175 seconds for LIMO, and 2187.5204 seconds for ContrastiveLIMO on one MI355X GPU each. These values do not identify a software stack.

### 7.3 Model inference and evaluation environments

| Campaign | Hardware and model | Runtime and engine settings | Request settings and evidence |
|---|---|---|---|
| G08 evaluation | One AMD Instinct MI355X GPU requested per Slurm array task; DeepSeek-R1-Distill-Qwen-7B base or merged SFT models. The evaluated merged checkpoint IDs and weight revisions are not retained in the public release. | The launch script runs vLLM with BF16, maximum model length 32,768, and GPU-memory utilization 0.85. Retained runtime output does not identify the resulting tensor-parallel or quantization settings, nor the exact vLLM, Python, PyTorch, Transformers, or HIP/ROCm versions for G08. | Temperature 0.6; top-p 0.95; maximum 16,384 generated tokens; client concurrency 8; request seed 0; request timeout 21,600 seconds. |
| G09 evaluation | One AMD Instinct MI355X GPU requested per Slurm array task; DeepSeek-R1-Distill-Qwen-7B base or merged SFT models. The evaluated merged checkpoint IDs and weight revisions are not retained in the public release. | Completed vLLM log records vLLM `0.11.2.dev673+g839868462`, ROCm backend, tensor parallel 1, BF16, quantization `None`, maximum model length 32,768, GPU-memory utilization 0.85, and engine seed 0. | Temperature 0.6; top-p 0.95; maximum 16,384 generated tokens; client concurrency 8; request seed 0; request timeout 21,600 seconds. |

The evaluation launch commands do not specify top-k, `max_num_seqs`, or `max_num_batched_tokens`; the retained logs do not record explicit values for those settings. G09 runtime logs do not include PyTorch, Python, Transformers, or HIP/ROCm package versions, and no matching package snapshot was found. G08 has launch configuration and saved evaluation outputs, but no retained serving log to identify its vLLM version or effective tensor-parallel and quantization settings. GPU resource requests and successful ROCm-backed execution identify the GPU class per task, but the retained G08/G09 logs do not include a device-name probe for each task.

vLLM generated the model predictions; it did not perform shortest-path correctness verification. Evaluation used saved prediction records, answer extraction, and deterministic CPU-based shortest-path verification/offline scoring as separate stages. The public PathFACT framework remains CPU-only.

### 7.4 Experiment-to-environment mapping

| Experiment group | Training or generation environment | Evaluation environment | Mapping limit |
|---|---|---|---|
| Student 7B production generation | GEN-S (one MI300X GPU; vLLM `0.11.2.dev673+g839868462`, ROCm backend). | Not applicable to generation. | Completed 2,200-case Student run is identified by runtime metadata. |
| Teacher 32B production generation | GEN-T300: one MI300X GPU per 552-case Gold or Contrastive batch. | Not applicable to generation. | Completed production batches are identified; exact mapping of every released Teacher target to a specific later generation wave is incomplete. |
| Later Teacher 32B wave | GEN-T355: four MI355X one-GPU shard jobs; two Gold and two Contrastive. | Not applicable to generation. | Completed shard run configurations and logs exist; not every released Teacher target is mapped to a shard in public metadata. |
| Original Vanilla-SFT, LIMO, ContrastiveLIMO | One MI355X GPU per method in the runtime-cost record; training software stack not captured for the canonical runs. | G08/G09 launched on one MI355X per array task. | The evaluation campaigns are linked to reported method/recipe conditions, but there is no portable run/checkpoint-to-output manifest. |
| TrueWarm0 methods and Vanilla early stopping | No per-run GPU or software record. | G08/G09 campaign settings are recorded; no checkpoint-specific server record is retained for each condition. | Do not assume these used the Original training stack or hardware. |
| Alternative 1 LIMO and ContrastiveLIMO | No per-run GPU or software record. | No completed three-method accuracy comparison is reported. | Downstream checkpoint and evaluation mapping is absent. |
| Single- and Dual-Ranking | No per-run GPU or software record. | No checkpoint-specific inference mapping is retained. | The saved evidence supports the ranking data/weights, not a complete training-to-evaluation trace. |

The exact per-run checkpoint IDs, canonical SFT best-validation-loss epochs, and checkpoint-selection rules remain absent.

### 7.5 Environment evidence and specific gaps

| Evidence source type | Settings established | Remaining information not captured |
|---|---|---|
| Student runtime manifest and completed vLLM log | Resolved 7B model revision; Python, PyTorch, HIP runtime, Transformers, vLLM, ROCm backend, and launch/request settings. | Tokenizer snapshot revision and tokenizer file hash. |
| Teacher MI300X run configurations and completed vLLM logs | Model family; vLLM version; engine and request settings; GPU allocation. | Python, PyTorch, Transformers, exact HIP/ROCm runtime, and immutable model/tokenizer revisions. |
| Teacher MI355X run configurations, frozen container identity, and completed vLLM logs | GPU shard allocation; exact runtime package versions tied to the same frozen container as the Student runtime manifest; engine/request settings. | Immutable Teacher model/tokenizer revisions. |
| Canonical SFT recipe summary and runtime-cost record | Reported recipe values, completion, one MI355X per Original method, and recorded durations. | Run-matched training command/config, framework and package versions, omitted optimizer/tokenization details, checkpoint IDs, and selection metadata. |
| G08/G09 launch scripts, G09 vLLM startup log, and saved outputs | MI355X GPU request, request-level decoding settings, G09 vLLM/engine configuration, completed predictions. | G08 exact runtime version; G08/G09 PyTorch, Transformers, and HIP/ROCm package snapshots; explicit top-k, max-num-seqs, and max-num-batched-tokens values. |

Hardware and software differences across experiments must not be interpreted as causes of accuracy differences without a controlled comparison. Documented historical environments do not establish complete independent reproducibility.

## 8. Provenance and limitations

- Canonical SFT recipe values are summary-level historical reports. The missing run-level settings and supporting execution-record classes are listed above; no framework default has been substituted for an absent run value.
- Historical checkpoint-to-training-file provenance is incomplete, particularly the exact Vanilla-SFT checkpoint-to-training-file SHA256 mapping.
- Publicly available records establish experiment-level method, recipe, and evaluation-view relationships, but not a complete portable mapping from each run to the exact evaluated checkpoint and saved output artifact.
- PathFACT does not publish full or adapter model weights, optimizer states, scheduler states, checkpoint archives, or other training-state artifacts. This absence limits direct reproduction of the reported model evaluations.
- GPU training and model inference were performed outside the public CPU-only framework and remain the researcher's responsibility in a compatible environment.
- Complete historical training reproduction has not been independently established. No framework defaults are substituted for missing records.

### Evidence records

- Current public method-based dataset paths: [Dataset Map](../DATASET_MAP.md) and the [Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT).
