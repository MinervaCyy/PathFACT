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

These method-based filenames are the current v0.3 public paths. The record condition values remain `vanilla`, `gold`, and `contrastive`; `gold` denotes ordinary LIMO supervision. Each released file contains explicit `prompt` and `completion` text. The public Vanilla files were inspected: 200 training and 44 validation completions contain the correct final answer only, with no deliberately inserted empty reasoning wrapper. The historical source-data hashes are `eeea26777d60964128aa0a5fe8aef5007a4e808e3ec02fdd5f175b9f2af855cb` (Train-200) and `ab009102cb2243abde2184a3ab0e9923f1c7a6a4bf8754510a78db23f9c44d0f` (Loss-Val-44). The exported public JSONL files use a different serialization and therefore have different byte-level file hashes; those hashes are not evidence of content mismatch. Exact tokenizer revision, trainer-side tokenization procedure, and sequence-length/truncation settings remain unestablished.

## 3. Original training configuration

The canonical configuration below combines the retained experiment summaries with matched original training scripts, completed-job records, and execution logs.

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
| Hardware allocation | 1 × AMD Instinct MI355X GPU, 8 CPU cores, 128 GB RAM per job |
| Training implementation | `FastLanguageModel.from_pretrained()`, `FastLanguageModel.get_peft_model()`, and TRL `SFTTrainer` |
| Software environment | Unsloth `2026.8.18`; TRL `0.24.0`; PyTorch `2.10.0+rocm7.1`; HIP `7.1.25424`; Transformers `4.57.6`; bitsandbytes `0.50.1`; FlashAttention `2.8.3`; ROCm module `rocm/7.14` |
| Optimizer and checkpointing | `adamw_8bit`; Unsloth gradient checkpointing |
| Execution checks | Frozen-input hash check, MI355X environment check, and run-complete check passed for each original run |
| Completion | Vanilla-SFT, LIMO, and ContrastiveLIMO completed |

The shared values apply to the canonical three-method comparison. Vanilla-SFT uses correct final answers only, LIMO uses ordinary Teacher reasoning and answers, and ContrastiveLIMO uses failure-conditioned corrective Teacher reasoning and answers. The completed cost record reports 191.9927 seconds for Vanilla-SFT, 768.1175 seconds for LIMO, and 2187.5204 seconds for ContrastiveLIMO. These are historical measurements, not general hardware benchmarks.

The original three jobs have recovered run-matched scripts, completed Slurm records, logs, and software-environment evidence. Their trainer runtimes are Vanilla-SFT **191.9927 s**, LIMO **768.1175 s**, and ContrastiveLIMO **2187.5204 s**; these are trainer-reported durations, not total job durations. The Vanilla run also records loss `0.6203672255078951`. A distinct earlier Vanilla experiment has a 188.3396-second trainer runtime and loss `0.44496760225544374`; it is not the canonical pure answer-only baseline and is excluded from this cost comparison. Exact initialized model and tokenizer revisions, some tokenizer/truncation and LoRA details, and portable checkpoint-to-output identities remain unavailable. The retained records do not establish every omitted low-level trainer parameter.

The canonical Vanilla-SFT checkpoint is associated in the historical research records with task-only correct-answer counts of 44 (G08 Balanced-300), 62 (G08 TrainDist-300), 196 (G09 Balanced-846), and 486 (G09 TrainDist-1600). Earlier documentation of an empty-`<think></think>` control describes a reasoning-suppression diagnostic; it is not the canonical pure answer-only baseline. The available records do not justify identifying the earlier `VanillaSFT_G06_sft_v1` experiment with that diagnostic.

## 4. Training-recipe sensitivity

The reported recipe differences are:

| Recipe | Methods / status | Learning rate | Effective batch | Epochs | Scheduler / warmup | Seed | Evidence boundary |
|---|---|---:|---:|---:|---|---:|---|
| Original | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 5e-5 | 64 | 15 | Linear; 10 warmup steps | 3407 | Run-matched scripts and execution evidence recovered; hardware, software, optimizer, and trainer implementation are documented above. |
| Alternative1 | Vanilla-SFT, LIMO, ContrastiveLIMO completed; no complete comparable three-method downstream accuracy evaluation established | 1e-3 | 32 | 5 | Linear; 2 warmup steps | 3407 | Completed training does not establish task-level evaluation. Runtime: 69.2571 s, 262.2280 s, 745.7256 s respectively. |
| Warm0 preliminary | Preliminary runs retained warmup steps | — | — | — | Warmup greater than zero | — | Distinct from corrected TrueWarm0; no settings or results are transferred between them. |
| TrueWarm0 | Completed Vanilla-SFT, LIMO, ContrastiveLIMO | 1e-4 | 64 | 15 | Cosine; exactly 0 warmup steps | 3407 | Runtime: 188.4309 s, 772.2450 s, 2190.7105 s respectively. |
| TrueWarm0 Vanilla E5 | Separate answer-only Vanilla run, stopped after five completed epochs | 1e-4 | 64 | 5 | Original 15-epoch scheduler horizon retained; exactly 0 warmup steps | 3407 | Separate early-stopped run, not an E5 checkpoint selected from a completed E15 run; runtime 67.3214 s. |

The Original-to-TrueWarm0 comparison changes learning rate and scheduler/warmup configuration together, so it does not isolate a single causal parameter. Alternative1's three completed training jobs do not establish a complete comparable three-method task-level evaluation. The early-stopped Vanilla run is documented separately from the 15-epoch run and is not an epoch-5 checkpoint selected from a completed E15 run. Alternative1, TrueWarm0, and Vanilla E5 logs report the original core software versions; all these jobs used one MI355X GPU, 8 CPU cores, and 128 GB RAM.

Historical experiment records report task-only evaluation across G08 Balanced, G08 TrainDist, G09 Balanced, and G09 TrainDist for Original and TrueWarm0 E15 models, plus the separate TrueWarm0 Vanilla E5 run. These evaluation labels and result aggregates do not identify exact checkpoint artifacts or portable run IDs. The retained records contain no canonical best-validation epoch or checkpoint-selection rule.

## 5. Run-to-evaluation traceability

Portable checkpoint artifacts and a complete run/checkpoint-to-output manifest are not public. The following table maps the reported method/recipe conditions to evaluation views and epochs recorded in the experiment summaries.

| Methods | Recipe | Evaluated epoch | Evaluation views | Evidence boundary |
|---|---|---:|---|---|
| Vanilla-SFT, LIMO, ContrastiveLIMO | Original | 15 (reported) | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Canonical Vanilla checkpoint is associated with correct-answer counts 44, 62, 196, and 486 on those four task-only views respectively; portable checkpoint ID and full output mapping are unavailable. |
| Vanilla-SFT, LIMO, ContrastiveLIMO | TrueWarm0 | 15 (reported) | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Condition and epoch are reported; exact checkpoint IDs and checkpoint-selection metadata are absent. |
| Vanilla-SFT | TrueWarm0 early stop | 5 completed | G08 Balanced, G08 TrainDist, G09 Balanced, G09 TrainDist; Original/Stay/Switch counterfactual evaluation | Separate five-epoch early-stopped run, not an E5 checkpoint selected from a completed E15 run. |
| Vanilla-SFT, LIMO, ContrastiveLIMO | Alternative1 | 5 completed | No complete comparable three-method task-level evaluation established | Training completion is verified; this is not evidence of downstream evaluation. |

The Original/Stay/Switch evaluation contains eight conditions: shared Base; the three Original E15 methods; the three TrueWarm0 E15 methods; and TrueWarm0 Vanilla E5. Condition-level aggregates are reported, but exact portable checkpoint identifiers and saved-result filenames are not present in the public release. No private filesystem paths are reproduced.

## 6. Ranking-supervision configurations

The earlier Single- and Dual-Ranking study augments **LIMO SFT**, not ContrastiveLIMO, with answer-level preference supervision. It uses G06 Train-200 and the 162 strict ranking cases; Single applies Switch ranking at weight 0.1, while Dual splits that weight across Original and Switch (0.05 each). Preserve this study's own historical configurations, evaluations, and negative findings; later ranking conditions do not retroactively define it.

Recovered execution evidence establishes MI355X training hardware for this earlier study, but the public trace still does not provide a complete portable mapping from every ranking checkpoint to evaluated outputs. Boundary-Local evidence remains separate; no Boundary-Local task-only VSP result is claimed.

### Confirmatory Ranking-185 experiments

These later runs use 200 training records (185 ranking-eligible, 15 SFT-only) and 44 loss-validation records. They are distinct from the earlier Single-/Dual-Ranking study and its 162 strict cases.

| Ranking design | Trainer runtime | Final-epoch validation loss |
|---|---:|---:|
| `switch_only_local` | 1749.3821 s | 0.22146011888980865 |
| `original_local_switch` | 2232.4151 s | 0.21112333238124847 |
| `boundary_paired` | 2239.8359 s | 0.20362186431884766 |

All three completed 15 epochs with seed 3407, ranking weight 0.1, and margin 1.0, using one MI355X GPU per job. Logs record Unsloth `2026.8.18`, Transformers `4.57.6`, PyTorch `2.10.0+rocm7.1`, ROCm toolkit `7.1.25424`, Triton `3.7.1`, and BF16. Frozen-data preflight and ranking-mathematics self-test passed. These final-epoch validation losses are not task-level shortest-path accuracy and do not replace selected confirmatory checkpoint metrics. The recovered evidence does not establish a consistent improvement over ordinary LIMO.

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
| Original Vanilla-SFT, LIMO, ContrastiveLIMO | 1 × AMD Instinct MI355X, 8 CPU cores, 128 GB RAM per job. | Unsloth `2026.8.18`; TRL `0.24.0`; PyTorch `2.10.0+rocm7.1`; HIP `7.1.25424`; Transformers `4.57.6`; bitsandbytes `0.50.1`; FlashAttention `2.8.3`; ROCm module `rocm/7.14`. Original implementation uses `FastLanguageModel.from_pretrained()`, `FastLanguageModel.get_peft_model()`, and TRL `SFTTrainer`; optimizer `adamw_8bit`; Unsloth gradient checkpointing. | Frozen input hash, MI355X environment, and completion checks passed. Trainer runtimes: 191.9927 s, 768.1175 s, and 2187.5204 s respectively. |
| Alternative1, TrueWarm0 E15, TrueWarm0 Vanilla E5 | 1 × AMD Instinct MI355X, 8 CPU cores, 128 GB RAM per job. | Alternative1 and TrueWarm0 logs report the same core software versions as the original SFT runs. | All listed training jobs completed; Alternative1 does not have a complete comparable three-method downstream accuracy evaluation. Preliminary Warm0 remains distinct from corrected TrueWarm0. |
| Earlier Single-/Dual-Ranking study | MI355X training hardware verified. | Recovered execution evidence exists; settings and outcomes remain those of the earlier study. | Keep its 162 strict cases and own preference-supervision configuration separate from Ranking-185. |
| Confirmatory Ranking-185 | 1 × AMD Instinct MI355X per job. | Unsloth `2026.8.18`; Transformers `4.57.6`; PyTorch `2.10.0+rocm7.1`; ROCm toolkit `7.1.25424`; Triton `3.7.1`; BF16. | Three completed runs; data preflight and ranking-mathematics self-test passed. Validation losses are not task accuracy. |

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
| Original Vanilla-SFT, LIMO, ContrastiveLIMO | Verified MI355X training environments and run-matched original software stack. | G08/G09 launched on one MI355X per array task. | Historical method/recipe evaluation links are known; portable checkpoint-to-output manifest and public weights are unavailable. |
| TrueWarm0 methods and Vanilla E5 | Verified MI355X allocation; original core software versions recovered. | G08/G09 campaign settings are recorded. | Per-condition portable checkpoint-to-output mapping remains incomplete. |
| Alternative1 methods | Verified MI355X allocation and core software versions; all three jobs completed. | No complete comparable three-method task-level evaluation established. | Training completion must not be read as evaluated task performance. |
| Earlier Single-/Dual-Ranking | MI355X training hardware verified. | Historical task-only evaluations are documented. | Its own settings and results remain distinct; no complete portable checkpoint-to-output manifest. |
| Confirmatory Ranking-185 | Verified MI355X training environment and completed logs. | Final-epoch validation losses are recorded; do not treat as task-level accuracy. | Exact portable checkpoint-to-evaluation mapping remains incomplete. |

The exact per-run checkpoint IDs, canonical SFT best-validation-loss epochs, and checkpoint-selection rules remain absent.

### 7.5 Environment evidence and specific gaps

| Evidence source type | Settings established | Remaining information not captured |
|---|---|---|
| Student runtime manifest and completed vLLM log | Resolved 7B model revision; Python, PyTorch, HIP runtime, Transformers, vLLM, ROCm backend, and launch/request settings. | Tokenizer snapshot revision and tokenizer file hash. |
| Teacher MI300X run configurations and completed vLLM logs | Model family; vLLM version; engine and request settings; GPU allocation. | Python, PyTorch, Transformers, exact HIP/ROCm runtime, and immutable model/tokenizer revisions. |
| Teacher MI355X run configurations, frozen container identity, and completed vLLM logs | GPU shard allocation; exact runtime package versions tied to the same frozen container as the Student runtime manifest; engine/request settings. | Immutable Teacher model/tokenizer revisions. |
| Original SFT scripts, completed-job records, logs, and environment evidence | MI355X allocations; core software versions; training implementation; optimizer/checkpointing; runtime; input/environment/completion checks. | Exact model/tokenizer revisions, some low-level trainer and serialization settings, checkpoint IDs, and full checkpoint-to-output mapping. |
| Sensitivity-job logs and execution records | MI355X allocation for Alternative1, TrueWarm0 and Vanilla E5; runtimes and core software versions. | Complete comparable Alternative1 task-level evaluation; portable per-checkpoint output mapping. |
| Ranking execution logs | MI355X evidence for earlier and Ranking-185 groups; Ranking-185 versions, runtimes, and final-epoch validation losses. | Complete portable mapping to every evaluated checkpoint/output; validation loss does not establish task accuracy. |
| G08/G09 launch scripts, G09 vLLM startup log, and saved outputs | MI355X GPU request, request-level decoding settings, G09 vLLM/engine configuration, completed predictions. | G08 exact runtime version; G08/G09 PyTorch, Transformers, and HIP/ROCm package snapshots; explicit top-k, max-num-seqs, and max-num-batched-tokens values. |

Hardware and software differences across experiments must not be interpreted as causes of accuracy differences without a controlled comparison. Documented historical environments do not establish complete independent reproducibility.

## 8. Provenance and limitations

- Original SFT runs are supported by matched scripts, completed-job records, logs, environment evidence, and input-integrity checks. This establishes execution settings and completion, but not all low-level trainer settings or exact model/tokenizer revisions.
- The public Vanilla files were content-checked as pure answer-only supervision. Historical source-input hashes and public exported JSONL byte hashes differ because the public files use a different serialization; this check does not establish byte identity between those representations.
- Historical checkpoint-to-training-file provenance and the portable mapping from every checkpoint to published inference outputs remain incomplete.
- Publicly available records establish experiment-level method, recipe, and evaluation-view relationships, but not a complete portable mapping from each run to the exact evaluated checkpoint and saved output artifact.
- PathFACT does not publish full or adapter model weights, optimizer states, scheduler states, checkpoint archives, or other training-state artifacts. This absence limits direct reproduction of the reported model evaluations.
- GPU training and model inference were performed outside the public CPU-only framework and remain the researcher's responsibility in a compatible environment.
- Complete end-to-end reproduction has not been established: public checkpoint weights and complete execution artifacts are unavailable, and exact model/tokenizer revisions and portable checkpoint-to-output mappings remain incomplete.

### Evidence records

- Current public method-based dataset paths: [Dataset Map](../DATASET_MAP.md) and the [Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT).
