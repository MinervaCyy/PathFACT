# PathFACT v0.1 Data Dictionary

Source attribution for examples: PathFACT Dataset v0.1, MinervaCYY/PathFACT, [fixed revision `3d20d452e1efe21c80d55185f5176ca3fc5e4f9f`](https://huggingface.co/datasets/MinervaCYY/PathFACT/tree/3d20d452e1efe21c80d55185f5176ca3fc5e4f9f). Dataset excerpts are subject to the release's limited CC BY 4.0 scope and exclusions described in the [Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT). The repository's MIT licence applies to software, not automatically to dataset content.

This describes the 13 released JSONL files in `pathfact-v0.1-hf-upload`, checked against every local row on 2026-11-04 and the implementation in `src/pathfact/datasets/public.py`, `derived.py`, `historical_vsp.py`, and `graphs/canonical.py`. JSON types below are observed types; `null` means JSON null. All listed ordinary fields were present on every row unless an optionality note says otherwise. Node IDs are graph-map keys, not schema fields.

## G00 task records: `tasks/g00_tasks.jsonl`

G00 contains 2,200 graph tasks. Each record has 31 top-level fields, grouped
below by what a reader uses them for. In the entries, **Original field** is the exact
JSON key, **Display name** is a reader-friendly label, and **JSON type** is
the type stored in the file. Arrays and objects contain the kinds of values
described in their explanations.

### A. Essential Task Information

These fields describe the problem and its answer. For a model solving a task,
use `topology` and `node_pair` as inputs. Treat the two reference fields as
answers for scoring; withhold them from a model during held-out evaluation.

| Original field | Display name | JSON type | Plain-English explanation | Interpretation or usage caveat |
|---|---|---|---|---|
| `topology` | Weighted graph | object | Maps each node ID to a list of its neighbors and the positive integer weight of each connecting edge. A pair such as `[1,160]` means an edge to node 1 with weight 160. | The graphs are undirected, so an edge is represented from both endpoints. This is task input. |
| `node_pair` | Start and end nodes | array of 2 integers | Gives the ordered endpoints: first the start, then the destination. | Reversing the order reverses the requested direction, even though the graph is undirected. This is task input. |
| `ground_truth_shortest_path` | Reference shortest route | integer array | Gives one route from the start to the destination that has the minimum possible total edge weight. | It is a reference answer, not model input during evaluation. Other routes can also be optimal when they have the same minimum cost. |
| `gold_cost` | Minimum route cost | integer | The total edge weight of a minimum-cost route for this task. | This is a reference answer; it measures summed weight, not the number of edges or hops. |
| `production_case_id` | Case ID | string | Identifies this task record and lets you match it to records for the same case in other PathFACT stages. | Use it as a record-linking key, not as a task instruction or a measure of difficulty. |

**Verified example — PR01233, cost 746.** This actual released row has six
nodes and endpoints `[0,5]`. Its only route is the chain shown; edge weights
160 + 121 + 190 + 108 + 167 add to 746.

```json
{"production_case_id":"PR01233","topology":{"0":[[1,160]],"1":[[0,160],[2,121]],"2":[[1,121],[3,190]],"3":[[2,190],[4,108]],"4":[[3,108],[5,167]],"5":[[4,167]]},"node_pair":[0,5],"ground_truth_shortest_path":[0,1,2,3,4,5],"gold_cost":746}
```

### B. Graph Characteristics

These fields summarize the graph and the reference route. They can help describe
or group tasks, but they are metadata about the task, not extra directions to a
model.

| Original field | Display name | JSON type | Plain-English explanation | Interpretation or usage caveat |
|---|---|---|---|---|
| `num_nodes` | Number of nodes | integer | Number of distinct nodes in the graph. | A graph-size description, not a difficulty score by itself. |
| `num_edges` | Number of edges | integer | Number of distinct connections in the undirected graph. | Each undirected connection is counted once, although it appears in both endpoint adjacency lists. |
| `topology_type` | Graph family | string | Names the family used to generate the graph, such as `fat_tree`, `barabasi_albert`, `erdos_renyi`, `grid`, or `jellyfish`. | The family is a category; its name is not an ordinal difficulty rating. |
| `topology_dimension` | Graph-family settings | object | Gives the parameter values that describe the chosen graph family. Keys can include `n`, `m`, `p`, `r`, `h`, or `max_degree`; for example, a fat-tree record can have `{"h":5,"r":1}`. | Which keys appear depends on the family, so records need not have identical keys. Values describe construction settings, not model performance. |
| `topology_dimension.n` | Node-count or grid-side setting | integer; present for applicable families | For Barabási–Albert, Erdős–Rényi, and jellyfish graphs, this is the total node count. For a square grid, it is the number of nodes on each side, so the graph has `n × n` nodes. | Interpret the setting together with `topology_type`; do not treat grid `n` as total node count. |
| `topology_dimension.m` | New-node attachment count | integer; present for Barabási–Albert graphs | Number of edges each new node attaches with in the Barabási–Albert generator. | A family-specific construction setting, not the graph's total edge count. |
| `topology_dimension.p` | Edge-probability setting | number; present for Erdős–Rényi graphs | Probability used by the Erdős–Rényi generator when considering possible edges. | A construction probability, not the observed fraction of edges. |
| `topology_dimension.h` | Fat-tree height | integer; present for fat-tree graphs | Height parameter passed to the balanced-tree generator used for this graph family. | Not the shortest route's hop count. |
| `topology_dimension.r` | Fat-tree branching factor | integer; present for fat-tree graphs | Branching-factor parameter passed to the balanced-tree generator. | Family-specific; it is not an endpoint or route value. |
| `topology_dimension.max_degree` | Jellyfish maximum degree | integer; present for jellyfish graphs | Maximum degree setting used by the jellyfish graph generator. | A construction limit, not necessarily the degree of every node. |
| `weight_distribution` | Edge-weight method | string | Names the recorded method used to assign edge weights. | It is the method label, not a summary statistic or the individual weights. |
| `gold_path_hops` | Hops in reference route | integer | Number of edges in `ground_truth_shortest_path`. | Hops count edges; they do not determine the minimum-cost objective. In this release, validation checks this equals the minimum hop count among optimal routes. |
| `hop_bucket` | Reference-route hop range | string | A category used to group the task by its recorded route-hop count, such as `1-2` or `5-8`. | It is a grouping label, not an exact hop count or universal difficulty score. |
| `min_optimal_hops` | Fewest hops among cheapest routes | integer | The smallest number of edges used by any minimum-cost route for the endpoint pair. | Among equally low-cost routes, this is the fewest-hop option. The released reference route uses this number of hops. |
| `max_optimal_hops` | Most hops among cheapest routes | integer | The largest number of edges used by any minimum-cost route for the endpoint pair. | It describes the range of hop counts among optimal routes; it does not change the minimum-cost objective. |
| `min_total_cost` | Recorded minimum cost check | integer | Repeats the task's minimum route cost. | The release validator checks it equals `gold_cost`; it is a duplicate check value, not a second cost threshold. |

### C. Advanced Generation and Sampling Metadata

These fields explain how graphs and endpoint pairs were generated, selected, or
identified. They are useful for provenance and sampling analysis. They are not
additional task inputs, and sampling values should not be added to a model's
prompt.

**Verified sampling example — PR00000.** The construction metadata records 43
eligible endpoint pairs in total: 20 in the selected one-hop category and 23
in the two-hop category. The selected exact-hop category is 1, and the chosen
endpoints are `[3,5]`. These counts describe candidate-pair selection; they do
not describe model input.

| Original field | Display name | JSON type | Plain-English explanation | Interpretation or usage caveat |
|---|---|---|---|---|
| `selected_exact_hop` | Chosen hop category | integer | The hop-count category selected when choosing an eligible endpoint pair. | It describes pair selection. In PR00000 the value is 1, matching the chosen pair's one-hop optimal route; it is metadata, not a request to the model. |
| `feasible_exact_hops` | Available hop categories | integer array | Lists the hop-count categories that had eligible endpoint pairs for selection. | The chosen value is recorded separately in `selected_exact_hop`; the list describes the candidate pool. |
| `eligible_pair_count_total` | Eligible pairs across categories | integer | Counts all endpoint pairs that met the task's pair-selection rules across the available categories. | A count from the construction pool, not the number of model attempts or answers. For PR00000 it is 43. |
| `eligible_pair_count_selected_exact_hop` | Eligible pairs in chosen category | integer | Counts eligible endpoint pairs in the chosen hop category. | This is the size of the selected category's candidate pool, not the number of samples ultimately released. For PR00000 it is 20; the other category contains 23. |
| `graph_seed` | Graph-generation seed | integer | Seed value recorded for generating the graph's structure. | Reproduction metadata; it does not itself guarantee that a separate software version will recreate identical bytes. |
| `weight_seed` | Edge-weight seed | integer | Seed value recorded for assigning edge weights. | Reproduction metadata, not an edge weight or task instruction. |
| `pair_seed` | Endpoint-selection seed | integer | Seed value recorded for choosing the endpoint pair. | Reproduction metadata for sampling, not a task input. |
| `generation_attempt_index` | Accepted attempt number | integer | Zero-based position of the generation attempt that produced this record. | Bookkeeping about generation, not a difficulty measure. |
| `attempts_to_accept` | Attempts before acceptance | integer | Number of generation attempts recorded before this task was accepted. | Bookkeeping about generation, not model effort or task difficulty. |
| `replicate` | Replicate index | integer | Identifies the recorded replicate within the generator's organization. | Provenance label, not a performance score. |
| `slot_id` | Generator slot ID | string | Names the generator slot associated with the record. | Opaque provenance identifier; do not interpret its text as a scientific score. |
| `support_row` | Support-table row | integer | Identifies a row in the supporting generation data used by the pipeline. | Provenance reference; it is not a row number in this JSONL file. |
| `pair_orientation_swapped` | Endpoint order reversed during selection | boolean | Records whether the selected endpoint order was reversed from the pair's canonical ordering. | `true` indicates a reversal in construction. Use `node_pair` itself as the authoritative task order. |
| `graph_structure_sha256` | Unweighted graph fingerprint | string (64-character hexadecimal) | A SHA-256 fingerprint identifying the graph structure without its edge weights. | Useful for identity checks; a hash alone does not establish correctness or reuse rights. |
| `weighted_graph_sha256` | Weighted graph fingerprint | string (64-character hexadecimal) | A SHA-256 fingerprint identifying the graph structure together with its edge weights. | Useful for identity checks; a hash alone does not establish correctness or reuse rights. |
| `case_instance_sha256` | Task-instance fingerprint | string (64-character hexadecimal) | A SHA-256 fingerprint identifying the weighted graph together with the ordered endpoints. | Endpoint order is part of the task identity. A hash alone does not establish correctness or reuse rights. |

The three groups above cover every field in the G00 records. Group A identifies
task inputs, reference answers, and the case-linking key; Group B describes the
graph and its optimal route; Group C records generation and sampling history.

## Genuine Student–Teacher pairs: G05

The released `paired/g05_paired.jsonl` contains 244 records. Each preserves a
`task`, the associated `student` record, and `teacher_targets` with separate
`gold` and `contrastive` objects. Field paths below are exact JSON paths.
Unless noted otherwise, every field is present and non-null in all 244 rows.
The tables give a reader-friendly name, JSON type/nullability, explanation,
research use, and an interpretation caveat for each field.

### How to Read a G05 Record

Read the record in this order: **task** (the graph, endpoints, and solver
reference) → **Student failure** (the submitted response, extracted route, and
verifier result) → **Gold and Contrastive Teacher targets** (two supervision
conditions for the same task) → **validation** (what was checked and what the
checks do not establish). The case ID links the paired record to its task and
other release stages.

### A. Task and Case Identity

The graph and ordered endpoints define the shortest-path problem. The
reference route and cost are solver answers, not Student answers.

| Exact original field path | Display name | JSON type and nullability | Explanation | Research use | Caveat |
|---|---|---|---|---|---|
| `schema_version` | Paired-record format | string; non-null | Format label `pathfact-paired-v0.1`. | Select a compatible reader. | Schema label, not a scientific condition. |
| `production_case_id` | Case ID | string; non-null | Identifies the paired case. | Join this record to the same case in other stages. | Identifier, not a task or model property. |
| `task` | Shortest-path task | object; non-null | Contains graph, endpoints, reference answer, and task metadata. | Use its graph and endpoints as the problem; use reference fields for scoring. | Preserves the G00 task as a nested object. |
| `task.production_case_id` | Task case ID | string; non-null | Case ID inside the task object. | Check task identity. | Matches the top-level case ID; not a second case. |
| `task.topology` | Weighted graph | object; non-null | Maps each node to its neighbors and positive edge weights. | Model/solver input. | Undirected edges appear in both endpoint adjacency lists. |
| `task.node_pair` | Start and destination | array of 2 integers; non-null | Ordered source and destination nodes. | Model/solver input. | Use this order when checking route endpoints. |
| `task.ground_truth_shortest_path` | Solver reference route | integer array; non-null | One route with minimum total edge weight. | Score candidate routes. | Different equal-cost routes may also be correct. This is not `student.predicted_path`; withhold it from evaluated models. |
| `task.gold_cost` | Minimum route cost | integer; non-null | Minimum possible sum of edge weights between the endpoints. | Compare a candidate route's cost with the optimum. | Cost, not hop count. |
| `validation_tier` | Pair validation tier | string; non-null | Summary derived from the two Teacher validation profiles. | Stratify analyses by recorded validation coverage. | See Section D for tier counts and limits; it is not a sentence-by-sentence guarantee. |
| `student` | Student record | object; non-null | Contains the Student response, failure checks, trace, and generation provenance. | Inspect the selected failure. | A wrong extracted answer does not make every sentence in its trace wrong. |
| `teacher_targets` | Teacher target pair | object; non-null | Contains separate `gold` and `contrastive` objects. | Compare the two supervision conditions for one case. | Keep the nested target conditions distinct. |
| `teacher_targets.gold` | Gold-condition target | object; non-null | Teacher target under the Gold condition. | Use as Gold supervision or comparison. | Not the Contrastive target. |
| `teacher_targets.contrastive` | Contrastive-condition target | object; non-null | Corrective Teacher target linked to the selected Student trace. | Study failure-conditioned corrective supervision. | Its trace-link check is distinct from factuality validation. |
| `joint_valid` | Pair linkage checks passed | boolean; non-null; `true` in all rows | Records that the task, genuine Student failure, and Teacher targets passed the joint checks. | Filter for pairs passing recorded cross-stage checks. | Does not certify every sentence in any reasoning text. |

### B. Student Failure Information

The Student's `final_content` is the submitted final response. The frozen
`reasoning_trace` is a separate field containing the selected reasoning text.
They can disagree: the reasoning may discuss or conclude one route while the
final answer submits another. The verifier extracts `predicted_path` from the
answer and checks it against `task.topology`, `task.node_pair`, and
`task.gold_cost`.

| Exact original field path | Display name | JSON type and nullability | Explanation | Research use | Caveat |
|---|---|---|---|---|---|
| `student.reasoning_trace` | Frozen Student reasoning trace | string; non-null | Selected canonical Student reasoning text. | Analyze reasoning associated with the failed answer or condition corrective supervision. | Not the final response and not evidence that every sentence is false. |
| `student.reasoning_trace_sha256` | Reasoning trace fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the exact UTF-8 trace. | Check trace text identity. | A hash does not establish correctness. |
| `student.final_content` | Student final response | string; non-null | Student's complete final response text. | Inspect how the Student presented its answer. | Distinct from the reasoning trace; verifier checks the extracted route separately. |
| `student.predicted_path` | Extracted Student route | integer array; non-null | Route extracted from the Student response. | Compare the Student answer with the solver reference. | Unlike the reference route, it may fail endpoints, edge continuity, or optimality. |
| `student.predicted_cost` | Student route cost | number or null; null in 138/244 | Sum of edge weights when every consecutive edge exists; otherwise unavailable and null. | Compare a computable route cost with `task.gold_cost`. | Null does not mean zero; cost alone does not establish a valid route. |
| `student.failure_category` | Verifier failure category | string; non-null | First applicable recorded failure category. Released values: `wrong_source_or_destination`, `nonexistent_edge`, `continuous_but_suboptimal`. | Group failures by verifier outcome. | Check outcome, not a complete diagnosis of the reasoning. |
| `student.error_type` | Source error annotation | string; non-null; empty in all rows | Preserved source annotation field. | Use only if comparing to a source where it is populated. | Empty in all 244 released rows; no further semantics are supported. |
| `student.syntactic_valid` | Route answer extracted in usable form | boolean; non-null | Whether the verifier could treat the extracted answer as a path value. | Diagnose extraction before graph checks. | A usable path value can still be wrong. |
| `student.path_continuity` | All route edges exist | boolean; non-null | Whether every consecutive node pair is an edge in the graph. | Distinguish missing-edge from cost failures. | Does not check endpoints or optimality. |
| `student.no_repeated_nodes` | No repeated route nodes | boolean; non-null | Whether the candidate route visits each node at most once. | Check a simple-path condition. | Does not establish correct endpoints, edges, or cost. |
| `student.source_target_consistent` | Correct endpoints | boolean; non-null | Whether the candidate starts and ends at the requested nodes. | Diagnose endpoint errors. | Does not check edges or optimality. |
| `student.vsp_valid` | Student route passes verifier | boolean; non-null; `false` in all rows | True only if extraction, endpoints, no repeats, edge continuity, and minimum cost all pass. | Identify failed extracted routes. | Applies to the final path, not every statement in the trace. |
| `student.answer_extracted` | Answer was extracted | boolean; non-null; `true` in all rows | Pipeline extracted a final route answer. | Confirm a candidate answer exists. | Extracted answers can still be invalid or suboptimal. |
| `student.reasoning_present` | Reasoning was present | boolean; non-null; `true` in all rows | Source response included reasoning text. | Filter for reasoning-bearing failures. | Presence does not imply truth or quality. |
| `student.stopped_normally` | Normal generation stop | boolean; non-null; `true` in all rows | Response was recorded as ending normally. | Exclude recorded abnormal stops. | Does not validate the answer. |
| `student.token_limit_truncation` | Token-limit truncation | boolean; non-null; `false` in all rows | Whether source marked the response as cut off by its token limit. | Check recorded completion status. | False does not establish correctness. |

An **invalid path** fails a structural check, such as having the wrong
endpoints or using an edge absent from the graph. A **valid-but-suboptimal
path** has the correct endpoints and existing edges, but costs more than the
minimum. The verifier checks extraction/syntax, endpoints, repeated nodes,
edge continuity, then optimal cost; `failure_category` records the first
applicable failure. The solver reference path and Student predicted path are
therefore different kinds of evidence: one is the minimum-cost label, the
other is the Student's candidate answer.

### C. Gold and Contrastive Teacher Targets

Gold is the Teacher target under the Gold condition. Contrastive is the
corrective target linked to the selected Student trace. These are different
supervision conditions even when both extract the same optimal final route:
the Gold target answers the task under Gold, while the Contrastive target is
conditioned on the failure and linked to its trace. Equal final routes do not
mean the target prompts, reasoning traces, or supervision roles are
interchangeable. Each target keeps its own nested object, answer, trace, and
metadata. No claim is made here that the two reasoning texts differ in any
particular way.

| Exact original field path | Display name | JSON type and nullability | Explanation | Research use | Caveat |
|---|---|---|---|---|---|
| `teacher_targets.gold.teacher_reasoning_trace` | Gold reasoning trace | string; non-null | Reasoning text associated with the Gold target. | Study or train on Gold-condition reasoning. | Interpret alongside its profile; not every sentence is thereby certified. |
| `teacher_targets.gold.teacher_reasoning_trace_sha256` | Gold trace fingerprint | 64-character hexadecimal string; non-null | SHA-256 of exact Gold trace text. | Check trace identity. | Hash is not a correctness judgment. |
| `teacher_targets.gold.teacher_final_content` | Gold final response | string; non-null | Teacher's final response for Gold condition. | Inspect or use the Gold completion. | Extracted route and its validity are recorded separately. |
| `teacher_targets.gold.teacher_final_content_sha256` | Gold response fingerprint | 64-character hexadecimal string; non-null | SHA-256 of exact Gold final-response text. | Check response identity. | Hash does not establish correctness. |
| `teacher_targets.gold.teacher_vsp_valid` | Gold answer passes path verifier | boolean; non-null; `true` in all rows | Extracted Gold route passes endpoint, path, and minimum-cost checks. | Filter for verified Gold final routes. | Final-path check; see Section D for its scope. |
| `teacher_targets.gold.validation_profile` | Gold validation profile | string; non-null | `vsp_plus_deterministic_rt_factuality_audit` or `solver_verified_vsp`. | Identify recorded Gold validation scope. | See Section D for tier counts and limits. |
| `teacher_targets.gold.model_label` | Gold model label | string; non-null | Source label for the Gold Teacher model. | Describe generation provenance. | Does not identify exact weights or revision. |
| `teacher_targets.gold.prompt_version` | Gold prompt version | string; non-null | Version label for the Gold Teacher prompt. | Track prompt provenance. | Prompt text is not included. |
| `teacher_targets.gold.prompt_sha256` | Gold prompt fingerprint | 64-character hexadecimal string; non-null | SHA-256 identifying the Gold prompt text. | Check prompt identity when prompt is available. | Hash does not reveal prompt content. |
| `teacher_targets.gold.tokenizer_protocol` | Gold tokenizer protocol | string; non-null | Label for tokenizer/chat-template protocol. | Track generation provenance. | Not a model-quality measure. |
| `teacher_targets.gold.generation_settings` | Gold generation settings | object; non-null | Contains numeric `temperature`, numeric `top_p`, integer `max_tokens`, integer `seed`. | Describe recorded sampling settings. | Provenance, not task input or proof of answer quality. |
| `teacher_targets.gold.predicted_path` | Gold extracted route | integer array; non-null | Route extracted from Gold final response. | Compare target answer with task reference. | A different equal-cost route can also be correct. |
| `teacher_targets.gold.predicted_cost` | Gold route cost | number; non-null | Cost computed for Gold extracted route. | Check against `task.gold_cost`. | Not an independent task label. |
| `teacher_targets.gold.stopped_normally` | Gold generation stopped normally | boolean; non-null | Recorded normal-stop status. | Check completion status. | Does not validate answer or reasoning. |
| `teacher_targets.gold.token_limit_truncation` | Gold token-limit truncation | boolean; non-null | Whether Gold response was recorded as truncated. | Check response completeness status. | No truncation does not validate reasoning. |
| `teacher_targets.contrastive.teacher_reasoning_trace` | Contrastive reasoning trace | string; non-null | Reasoning text associated with the corrective target. | Study or train on corrective reasoning. | Not every sentence is thereby certified. |
| `teacher_targets.contrastive.teacher_reasoning_trace_sha256` | Contrastive trace fingerprint | 64-character hexadecimal string; non-null | SHA-256 of exact Contrastive trace text. | Check trace identity. | Hash is not a factuality judgment. |
| `teacher_targets.contrastive.teacher_final_content` | Contrastive final response | string; non-null | Teacher's final response under Contrastive condition. | Inspect or use corrective completion. | Answer validity and reasoning validation are separate. |
| `teacher_targets.contrastive.teacher_final_content_sha256` | Contrastive response fingerprint | 64-character hexadecimal string; non-null | SHA-256 of exact final-response text. | Check response identity. | Hash does not establish correctness. |
| `teacher_targets.contrastive.teacher_vsp_valid` | Contrastive answer passes path verifier | boolean; non-null; `true` in all rows | Extracted Contrastive route passes endpoint, path, and minimum-cost checks. | Filter for verified corrective answers. | Final-path check; see Section D for its scope. |
| `teacher_targets.contrastive.validation_profile` | Contrastive validation profile | string; non-null | `vsp_plus_corrective_rt_factuality_validation` or `solver_verified_vsp`. | Identify recorded Contrastive validation scope. | See Section D for tier counts and limits. |
| `teacher_targets.contrastive.model_label` | Contrastive model label | string; non-null | Source label for the Contrastive Teacher model. | Describe generation provenance. | Does not identify exact weights or revision. |
| `teacher_targets.contrastive.prompt_version` | Contrastive prompt version | string; non-null | Version label for its prompt. | Track prompt provenance. | Prompt text is not included. |
| `teacher_targets.contrastive.prompt_sha256` | Contrastive prompt fingerprint | 64-character hexadecimal string; non-null | SHA-256 identifying the Contrastive prompt. | Check prompt identity when prompt is available. | Hash does not reveal prompt content. |
| `teacher_targets.contrastive.tokenizer_protocol` | Contrastive tokenizer protocol | string; non-null | Label for tokenizer/chat-template protocol. | Track generation provenance. | Not a model-quality measure. |
| `teacher_targets.contrastive.generation_settings` | Contrastive generation settings | object; non-null | Contains numeric `temperature`, numeric `top_p`, integer `max_tokens`, integer `seed`. | Describe recorded sampling settings. | Provenance, not task input or proof of quality. |
| `teacher_targets.contrastive.predicted_path` | Contrastive extracted route | integer array; non-null | Route extracted from Contrastive final response. | Compare corrective answer with reference. | A different equal-cost route can be correct. |
| `teacher_targets.contrastive.predicted_cost` | Contrastive route cost | number; non-null | Cost computed for extracted route. | Check against `task.gold_cost`. | Not an independent task label. |
| `teacher_targets.contrastive.stopped_normally` | Contrastive generation stopped normally | boolean; non-null | Recorded normal-stop status. | Check completion status. | Does not validate answer or reasoning. |
| `teacher_targets.contrastive.token_limit_truncation` | Contrastive token-limit truncation | boolean; non-null | Whether response was recorded as truncated. | Check response completeness. | No truncation does not validate reasoning. |
| `teacher_targets.contrastive.wrong_rt_continuity_verified` | Student-trace linkage verified | boolean; non-null; `true` in all rows | Records that Contrastive target passed the source's continuity-linkage check against the supplied Student trace. | Confirm the corrective target is linked to the intended failure. | Linkage check, not a final-path or reasoning-factuality check. |

### D. Validation and Scientific Limitations

The final-path verifier checks that a route was extracted, has the correct
endpoints, contains no repeated nodes, uses existing edges, and has minimum
cost. Thus `student.vsp_valid`
describes the submitted Student route, and each Teacher's `teacher_vsp_valid`
describes that Teacher's extracted final route. These booleans do not certify
the reasoning text.

The `validation_profile` fields specify the recorded reasoning-validation
tier. The verified distinction is **141/244 pairs with additional reasoning
factuality profiles for both Teacher targets** and **103/244 pairs with
solver-verified final-path checks only**. This additional validation must not
be described as independent verification of every Teacher reasoning sentence.
`validation_tier` summarizes those paired profiles as
`both_reasoning_factuality_validated` or `both_vsp_only`.

`teacher_targets.contrastive.wrong_rt_continuity_verified` reports a
continuity-linkage check between the Contrastive target and the supplied
Student trace. It is not a final-path or reasoning-factuality validation. All 244 Student
answers fail final-path verification, both Teacher answers pass it, and all
pairs are marked `joint_valid`. `error_type` is empty throughout; `selection_reason` is
preserved as a label and does not support diagnoses beyond its stored value.

**Concise genuine example — `PR01757`.** The full graph is present in the
record; omitted below to keep this example short.

```json
{
  "production_case_id": "PR01757",
  "task": {"node_pair": [3, 2], "ground_truth_shortest_path": [3, 8, 0, 7, 2], "gold_cost": 1300},
  "student": {"predicted_path": [3, 7, 2], "predicted_cost": 10100, "failure_category": "continuous_but_suboptimal", "vsp_valid": false, "answer_extracted": true, "reasoning_present": true},
  "teacher_targets": {
    "gold": {"predicted_path": [3, 8, 0, 7, 2], "predicted_cost": 1300, "teacher_vsp_valid": true},
    "contrastive": {"predicted_path": [3, 8, 0, 7, 2], "predicted_cost": 1300, "teacher_vsp_valid": true, "wrong_rt_continuity_verified": true}
  },
  "validation_tier": "both_reasoning_factuality_validated"
}
```

The Student route uses existing edges and correct endpoints but costs more
than the optimum. Both Teacher routes pass the final-path checks. Their longer reasoning texts
are omitted; their traces, hashes, and validation profiles remain in the
released record.

### E. Advanced Generation and Provenance Metadata

The task fields below are carried over from G00. They describe construction,
sampling, and identity, not additional model inputs. See the [G00 section](#g00-task-records-tasksg00_tasksjsonl)
for fuller explanations of the task metadata.

| Exact original field path | Display name | JSON type and nullability | Explanation | Research use | Caveat |
|---|---|---|---|---|---|
| `task.num_nodes` | Node count | integer; non-null | Number of graph nodes. | Describe graph size. | Not a standalone difficulty measure. |
| `task.num_edges` | Edge count | integer; non-null | Number of distinct undirected edges. | Describe graph size or density. | Each edge is listed at both endpoints but counted once. |
| `task.topology_type` | Graph family | string; non-null | Generator family label. | Group by graph family. | Category, not difficulty score. |
| `task.topology_dimension` | Family settings | object; non-null | Family-specific construction parameters. | Understand recorded graph setup. | Keys vary by family. |
| `task.topology_dimension.n` | Node-count or grid-side parameter | integer; present when applicable | For Barabási–Albert, Erdős–Rényi, and jellyfish graphs, `n` is the total node count. For a square grid, `n` is the number of nodes on each side, so the graph has `n × n` nodes in total. | Interpret or reproduce graph configuration. | Meaning depends on `task.topology_type`; do not treat grid `n` as its total node count. |
| `task.topology_dimension.m` | Attachment parameter | integer; present when applicable | Barabási–Albert parameter for the number of edges each new node attaches with. | Interpret or reproduce that graph family. | Family-specific; absent from other families. |
| `task.topology_dimension.p` | Edge-probability parameter | number; present when applicable | Erdős–Rényi probability used when generating possible edges. | Interpret or reproduce that graph family. | A construction probability, not an observed graph statistic. |
| `task.topology_dimension.h` | Tree-height parameter | integer; present when applicable | Fat-tree balanced-tree height setting. | Interpret or reproduce that graph family. | Not a shortest-path hop count. |
| `task.topology_dimension.r` | Tree-branching parameter | integer; present when applicable | Fat-tree branching-factor setting. | Interpret or reproduce that graph family. | Not an endpoint or route value. |
| `task.topology_dimension.max_degree` | Maximum-degree parameter | integer; present when applicable | Jellyfish-family maximum node-degree setting. | Interpret or reproduce that graph family. | A construction limit, not necessarily every node's degree. |
| `task.weight_distribution` | Weight-generation method | string; non-null | Recorded method used to assign edge weights. | Group by construction method. | Method label, not a statistic of actual weights. |
| `task.gold_path_hops` | Reference-route hops | integer; non-null | Edge count in the reference route. | Describe route length. | Objective is cost, not hops. |
| `task.hop_bucket` | Hop grouping | string; non-null | Category used to group tasks by hops. | Inspect grouping strata. | Not an exact hop count or difficulty score. |
| `task.selected_exact_hop` | Selected hop category | integer; non-null | Hop stratum selected during endpoint sampling. | Analyze sampling. | Metadata, not model instruction. |
| `task.min_optimal_hops` | Fewest optimal-route hops | integer; non-null | Fewest edges among minimum-cost routes. | Describe optimal-route lengths. | Does not replace the cost objective. |
| `task.max_optimal_hops` | Most optimal-route hops | integer; non-null | Most edges among minimum-cost routes. | Describe the range of optimal-route lengths. | Does not change the minimum cost. |
| `task.min_total_cost` | Duplicate minimum-cost check | integer; non-null | Duplicates `task.gold_cost`. | Check label agreement. | Validator requires equality; not a second threshold. |
| `task.feasible_exact_hops` | Available hop categories | integer array; non-null | Categories with eligible pairs in the sampling pool. | Inspect candidate strata. | Sampling metadata, not model input. |
| `task.eligible_pair_count_total` | Eligible pairs overall | integer; non-null | Count of endpoint pairs meeting selection rules across categories. | Analyze sampling-pool size. | Not model attempts or released-row count. |
| `task.eligible_pair_count_selected_exact_hop` | Eligible pairs in chosen category | integer; non-null | Candidate-pair count in the selected hop category. | Analyze chosen stratum size. | Not the number actually released. |
| `task.graph_seed` | Graph seed | integer; non-null | Seed recorded for graph structure generation. | Trace construction provenance. | Seed alone may not reproduce output across software versions. |
| `task.weight_seed` | Weight seed | integer; non-null | Seed recorded for edge-weight generation. | Trace weight provenance. | Not an edge weight. |
| `task.pair_seed` | Pair-selection seed | integer; non-null | Seed recorded for endpoint-pair selection. | Trace sampling provenance. | Not model input. |
| `task.generation_attempt_index` | Accepted attempt index | integer; non-null | Zero-based index of the attempt that produced the task. | Inspect construction history. | Bookkeeping, not difficulty. |
| `task.attempts_to_accept` | Attempts to acceptance | integer; non-null | Number of generation attempts recorded before acceptance. | Inspect construction history. | Not model effort. |
| `task.replicate` | Replicate identifier | integer; non-null | Generator replicate label. | Trace provenance. | Not a score. |
| `task.slot_id` | Generator slot identifier | string; non-null | Slot label associated with generation. | Trace provenance. | Opaque identifier, not a score. |
| `task.support_row` | Supporting-row identifier | integer; non-null | Row identifier in supporting generation data. | Trace provenance. | Not a row number in this JSONL. |
| `task.pair_orientation_swapped` | Endpoint orientation flag | boolean; non-null | Whether selected endpoint order was reversed from canonical order. | Understand endpoint sampling. | `task.node_pair` is authoritative. |
| `task.graph_structure_sha256` | Structure fingerprint | 64-character hexadecimal string; non-null | SHA-256 identity for unweighted graph structure. | Compare graph identity. | Hash alone proves neither correctness nor rights. |
| `task.weighted_graph_sha256` | Weighted graph fingerprint | 64-character hexadecimal string; non-null | SHA-256 identity for graph structure and weights. | Compare weighted graph identity. | Hash alone proves neither correctness nor rights. |
| `task.case_instance_sha256` | Task fingerprint | 64-character hexadecimal string; non-null | SHA-256 identity for weighted graph and ordered endpoints. | Compare task identity across stages. | Endpoint order is included; hash alone proves neither correctness nor rights. |
| `student.model_label` | Student model label | string; non-null | Source label for the Student model. | Describe generation provenance. | Does not identify exact weights/revision. |
| `student.prompt_version` | Student prompt version | string; non-null | Student prompt version label. | Track prompt provenance. | Prompt text is not included. |
| `student.prompt_sha256` | Student prompt fingerprint | 64-character hexadecimal string; non-null | SHA-256 identifying the Student prompt. | Check prompt identity if source prompt is available. | Does not reveal prompt text. |
| `student.tokenizer_protocol` | Student tokenizer protocol | string; non-null | Label for tokenizer/chat-template protocol. | Track generation provenance. | Not a capability measure. |
| `student.generation_settings` | Student generation settings | object; non-null | Contains numeric `temperature`, numeric `top_p`, integer `max_tokens`, integer `seed`. | Describe recorded sampling setup. | Metadata, not task input. |
| `student.generation_settings.temperature` | Student temperature | number; non-null | Recorded Student sampling temperature. | Describe generation settings. | Metadata, not task difficulty. |
| `student.generation_settings.top_p` | Student top-p setting | number; non-null | Recorded nucleus-sampling probability setting. | Describe generation settings. | Metadata, not task input. |
| `student.generation_settings.max_tokens` | Student token limit | integer; non-null | Maximum generation-token setting recorded for Student. | Describe generation limits. | Does not say how many tokens were actually used. |
| `student.generation_settings.seed` | Student generation seed | integer; non-null | Seed recorded for Student generation. | Trace generation provenance. | Does not guarantee identical output across systems. |
| `student.selection_reason` | Student selection label | string; non-null | Source label for why the selected record entered this subset. | Reproduce recorded selection grouping. | Values are `single_tier_a_attempt` (243) and `explicit_multi_attempt_resolution_for_downstream_continuity` (1); do not infer more than the label states. |
| `teacher_targets.gold.model_label` | Gold model label | string; non-null | Source label for Gold Teacher model. | Describe generation provenance. | Does not identify exact weights/revision. |
| `teacher_targets.gold.prompt_version` | Gold prompt version | string; non-null | Gold prompt version label. | Track prompt provenance. | Prompt text is not included. |
| `teacher_targets.gold.prompt_sha256` | Gold prompt fingerprint | 64-character hexadecimal string; non-null | SHA-256 identifying Gold prompt. | Check identity if prompt is available. | Does not reveal prompt text. |
| `teacher_targets.gold.tokenizer_protocol` | Gold tokenizer protocol | string; non-null | Label for tokenizer/chat-template protocol. | Track generation provenance. | Not a capability measure. |
| `teacher_targets.gold.generation_settings` | Gold generation settings | object; non-null | Contains numeric `temperature`, numeric `top_p`, integer `max_tokens`, integer `seed`. | Describe target sampling setup. | Metadata, not task input or proof of quality. |
| `teacher_targets.gold.generation_settings.temperature` | Gold temperature | number; non-null | Recorded Gold sampling temperature. | Describe generation settings. | Metadata, not answer validity. |
| `teacher_targets.gold.generation_settings.top_p` | Gold top-p setting | number; non-null | Recorded Gold nucleus-sampling probability setting. | Describe generation settings. | Metadata, not task input. |
| `teacher_targets.gold.generation_settings.max_tokens` | Gold token limit | integer; non-null | Maximum generation-token setting recorded for Gold. | Describe generation limits. | Does not say how many tokens were used. |
| `teacher_targets.gold.generation_settings.seed` | Gold generation seed | integer; non-null | Seed recorded for Gold generation. | Trace generation provenance. | Does not guarantee identical output across systems. |
| `teacher_targets.contrastive.model_label` | Contrastive model label | string; non-null | Source label for Contrastive Teacher model. | Describe generation provenance. | Does not identify exact weights/revision. |
| `teacher_targets.contrastive.prompt_version` | Contrastive prompt version | string; non-null | Contrastive prompt version label. | Track prompt provenance. | Prompt text is not included. |
| `teacher_targets.contrastive.prompt_sha256` | Contrastive prompt fingerprint | 64-character hexadecimal string; non-null | SHA-256 identifying Contrastive prompt. | Check identity if prompt is available. | Does not reveal prompt text. |
| `teacher_targets.contrastive.tokenizer_protocol` | Contrastive tokenizer protocol | string; non-null | Label for tokenizer/chat-template protocol. | Track generation provenance. | Not a capability measure. |
| `teacher_targets.contrastive.generation_settings` | Contrastive generation settings | object; non-null | Contains numeric `temperature`, numeric `top_p`, integer `max_tokens`, integer `seed`. | Describe target sampling setup. | Metadata, not task input or proof of quality. |
| `teacher_targets.contrastive.generation_settings.temperature` | Contrastive temperature | number; non-null | Recorded Contrastive sampling temperature. | Describe generation settings. | Metadata, not answer validity. |
| `teacher_targets.contrastive.generation_settings.top_p` | Contrastive top-p setting | number; non-null | Recorded Contrastive nucleus-sampling probability setting. | Describe generation settings. | Metadata, not task input. |
| `teacher_targets.contrastive.generation_settings.max_tokens` | Contrastive token limit | integer; non-null | Maximum generation-token setting recorded for Contrastive. | Describe generation limits. | Does not say how many tokens were used. |
| `teacher_targets.contrastive.generation_settings.seed` | Contrastive generation seed | integer; non-null | Seed recorded for Contrastive generation. | Trace generation provenance. | Does not guarantee identical output across systems. |

Prompt hashes and labels describe provenance; G05 does not include the original
prompt text. Interpret target text together with its validation fields and
their stated limits.

## G06 prompt/completion resources (four files)

G06 contains four frozen supervised fine-tuning (SFT) files: Gold Train-200,
Contrastive Train-200, Gold LossVal-44, and Contrastive LossVal-44. Each record
is one prompt paired with a target completion. The two conditions share case
IDs, so they are paired views of the same tasks rather than independent task
samples.

### A. What Is a G06 Training Example?

A G06 record gives a model a **prompt** and the **completion** it is trained to
produce. The prompt is the input; the completion is the target text, formatted
with a reasoning block and a verified shortest-path answer. G06 is a frozen
historical SFT resource: it contains examples, not model weights, trained
checkpoints, or observed model performance.

G06 records are linked to G05 by `production_case_id`. The task prompt comes
from the task-only prompt source. The Gold completion uses the matched G05
Gold Teacher reasoning trace; the Contrastive completion uses the matched G05
Contrastive Teacher reasoning trace. In both conditions the answer path is the
verified G00 shortest path for that case.

### B. Understanding Gold and Contrastive Prompts

The **Gold prompt** presents the graph task and asks for a shortest path. The
**Contrastive prompt** presents the same task together with the frozen
incorrect Student reasoning trace and corrective instructions. The Contrastive
prompt includes the exact G05 `student.reasoning_trace` text; this is the
Student's reasoning text, not the Student's submitted final answer. The latter
is stored separately in G05 as `student.final_content` (with its extracted
route in `student.predicted_path`). The final answer need not be inferred from
the reasoning trace.

Gold and Contrastive are different supervision conditions even when their
target completions end with the same optimal route. Gold trains on a response
to the task prompt. Contrastive trains on a response conditioned on the
Student's frozen error trace and corrective instructions. The task and final
route may therefore be shared while the input context and target reasoning
source differ. Identical case IDs identify the same underlying graph task,
not independent tasks.

The G06 builder constructs Gold prompts from task-only user messages and
Contrastive prompts from the matched corrective Teacher messages. It checks
that the full frozen Student trace occurs in the Contrastive message before
serializing it. Completions wrap the matched Teacher trace in `<think>` tags
and append `Answer: [verified path]`.

### C. Field-by-Field Data Dictionary

Every row in all four files has the following six fields. All are present and
non-null in the released records.

| Original field | Human-readable display name | JSON type | Plain-English explanation | Research use and interpretation caveat |
|---|---|---|---|---|
| `production_case_id` | Case ID | string | Identifies the graph task paired with this example. | Join to the matching G05/G00 case and align Gold with Contrastive. A shared ID means the same task, not an independent example. |
| `condition` | Supervision condition | string | Either `gold` or `contrastive`, naming the target condition. | Separate or compare the two SFT conditions. It is not a split label; Train versus LossVal is identified by the file. |
| `prompt` | Model input prompt | string | Frozen input text, including the graph/task request; Contrastive prompts also include the frozen incorrect Student reasoning trace and corrective instructions. | Feed this text as model input. Do not confuse the included Student reasoning in Contrastive with the Student's submitted final answer, or with the Teacher completion. |
| `prompt_sha256` | Prompt text fingerprint | string (64-character hexadecimal SHA-256) | Hash of the exact UTF-8 prompt string. | Check that prompt bytes/text have not changed. A hash is an integrity check, not a model-quality measure. |
| `completion` | Target completion | string | Frozen SFT target: Teacher reasoning text followed by the verified answer path. | Use as the target for supervised fine-tuning or loss calculation. It is a target, not an observed output from a model trained on G06. |
| `completion_sha256` | Completion text fingerprint | string (64-character hexadecimal SHA-256) | Hash of the exact UTF-8 completion string. | Check that target text has not changed. A hash does not validate the reasoning itself. |

### D. Training and Validation Splits

- The two Train files each contain **200 cases**; their case-ID sets match.
- The two LossVal files each contain **44 cases**; their case-ID sets match.
- Train and LossVal are disjoint and together cover the 244 G05 parent cases.
- Both conditions use the same partition: a case is in Gold Train and
  Contrastive Train, or in Gold LossVal and Contrastive LossVal.

LossVal-44 is held out from the Train-200 examples for loss validation during
SFT work. It is not a held-out task-solving evaluation: its prompt is paired
with a target completion, and it is intended for measuring loss against that
target. G08/G09 are the separate task-solving evaluation resources; their
reference answers should be withheld from models during evaluation.

The frozen split is stratified by shortest-path hop difficulty in the
construction code. Train quotas are 110 Easy, 75 Medium, and 15 Hard cases;
LossVal quotas are 24 Easy, 16 Medium, and 4 Hard. These strata describe the
historical G06 construction and are not model performance results.

### E. Scientific Limitations and Provenance

The four files are prompt/completion data only. They contain no model weights,
trained checkpoints, or outcomes from training or inference. A low loss on
LossVal would measure agreement with these frozen completions, not by itself
prove shortest-path competence or generalization. G08/G09 address task-solving
evaluation separately.

G06 derives its task identities and Teacher targets from the matched G05
records and uses the G00 verified reference path for completion answers. The
Contrastive prompt's embedded Student text is the frozen incorrect reasoning
trace; it is not the Student's separate submitted final answer. The two
conditions share cases and answer paths, so do not count Gold and Contrastive
rows as twice as many independent tasks. Prompt and completion hashes support
exact-text integrity checks, not scientific validation of all reasoning
sentences.

**Worked example — `PR01727` (Train-200).** This genuine case appears in both
conditions. The excerpts below are from the released prompts; the bracketed
omission note is editorial, and the long Student trace is not reproduced.

- **Case and G05 parent:** task endpoints `[6,3]`; reference route
  `[6,0,5,3]`, cost 2100. G05 records the Student's submitted answer as
  `[6,4,5,3]`, categorized `nonexistent_edge`; its predicted cost is null
  because the route contains a missing edge. G05 stores the reasoning trace
  separately from that final answer.
- **Gold prompt structure:** begins with the actual task instruction
  `You are a helpful assistant that solves shortest path routing problems in computer networks.`
  It ends with `Node pair: [6, 3]` before the assistant-response boundary.
- **Contrastive prompt structure:** includes the graph task, followed by the
  actual marker `STUDENT WRONG REASONING TRACE:` and the exact frozen G05
  Student reasoning text (omitted here), then corrective instructions. The
  released prompt ends with `End with the correct final path in the same
  answer format requested by the original task.`
- **Target completion relationship:** both released completions end with the
  verified answer `Answer: [6, 0, 5, 3]`. The Gold completion uses the G05 Gold
  Teacher trace; the Contrastive completion uses the G05 Contrastive Teacher
  trace. This example does not require a claim about how the reasoning texts
  differ.

## G08/G09 evaluation resources (four files)

The release contains four task-evaluation views: G08 Balanced-300,
G08 TrainDist-300, G09 Balanced-846, and G09 TrainDist-1600. Each record
contains a task input, solver reference labels, view membership, and graph/task
provenance. It does **not** contain a model-generated prediction or a model
performance result.

### A. What Are G08 and G09 Evaluation Views?

G08 and G09 are evaluation stages built from the G00 task universe. G00 has
2,200 task cases, including the 244 cases used for G06. The G09 source pool
contains the other 1,956 G00 cases (`G00 − G06`). G09 Balanced and TrainDist
are two overlapping views of that pool. The G07 candidate set contains
717 cases; its 300-case Balanced view, 300-case TrainDist view, and
Diagnostic-117 make a disjoint partition. Only the two 300-case G08 views are
published here; Diagnostic-117 is excluded. G07 is a subset of the G09 pool.

The release adapter records each row's source stage and view name, and checks
the upstream G08 `g08_dataset` label. G08 source records supply a `difficulty`
label; G09 source records are task records without a view or difficulty field,
so the adapter assigns their view name from the source file. The observed
compositions and limits of these views are detailed in Section D; their names
alone do not define an undocumented sampling algorithm.

### B. Model Inputs and Reference Answers

Use `model_input.topology` and `model_input.node_pair` as the model's task.
The published JSONL also exposes `reference.ground_truth_shortest_path` and
`reference.gold_cost`. Those are solver answers for scoring: **do not send
`reference` to the model** during an evaluation. The `provenance` object
describes graph construction, route properties, sampling, and fingerprints;
it is metadata, not part of the task prompt. In particular,
`provenance.min_total_cost` duplicates `reference.gold_cost`; withhold
provenance as well as reference labels from the model.

### C. Field-by-Field Data Dictionary

All top-level fields and all nested objects below are present and non-null in
every released row, except `view.source_difficulty`, which is present and
non-null in G08 and absent (not null) in G09, and the family-dependent keys
inside `provenance.topology_dimension`, which are omitted when not applicable.
Graph node IDs are dynamic keys in `model_input.topology`, not additional
schema fields.

#### Record fields

| Exact original field path | Human-readable display name | JSON type and optionality | Plain-English explanation | When to use it and caveat |
|---|---|---|---|---|
| `schema_version` | Evaluation-record format | string; required, non-null | Format identifier `pathfact-evaluation-v0.1`. | Select a compatible reader. It is a schema label, not an evaluation split. |
| `production_case_id` | Case ID | string; required, non-null | Stable identifier for this graph task. | Join the same case across stages and count unique tasks. Repeated membership in views does not create a new task. |
| `view` | Evaluation-view information | object; required, non-null | Stores source stage and view label, plus the G08 source difficulty when supplied. | Identify which evaluation view contains the record. View membership is not an answer or model result. |
| `model_input` | Model-facing task | object; required, non-null | Holds only the graph and ordered endpoint pair. | Provide this object to the model. Keep `reference` separate. |
| `reference` | Solver reference answer | object; required, non-null | Holds a minimum-cost route and its total cost. | Score a model's predicted route. Do not include it in model input. |
| `provenance` | Graph and task metadata | object; required, non-null | Holds graph size/family, route and sampling metadata, seeds, IDs, and fingerprints. | Audit or describe task construction. It is not extra model input or observed performance. |

#### View fields

| Exact original field path | Human-readable display name | JSON type and optionality | Plain-English explanation | When to use it and caveat |
|---|---|---|---|---|
| `view.source_stage` | Source stage | string; required, non-null | `G08` or `G09`, identifying the stage that selected the view. | Group records by source stage. It does not describe model performance. |
| `view.name` | View name | string; required, non-null | `balanced` or `train_distribution`, preserving the source-defined view label. | Select or report a view. Interpret alongside the verified row composition in Section D; do not infer undisclosed selection rules from the name alone. |
| `view.source_difficulty` | Source difficulty category | string; optional by stage; non-null when present | G08 source label `Easy`, `Medium`, or `Hard`. It is included only when the source provided it. | Describe G08's recorded path-difficulty strata. It is absent from all G09 rows; difficulty is not model performance. |

#### Model-input fields

| Exact original field path | Human-readable display name | JSON type and optionality | Plain-English explanation | When to use it and caveat |
|---|---|---|---|---|
| `model_input.topology` | Weighted graph | object; required, non-null | Adjacency map from node IDs to neighbor/weight pairs. | This is the graph supplied to the model. The graph is undirected; each edge is listed at both endpoints. |
| `model_input.node_pair` | Start and destination | array of 2 integers; required, non-null | Ordered source and destination nodes. | Supply these endpoints with the graph. Preserve their order. |

#### Reference fields

| Exact original field path | Human-readable display name | JSON type and optionality | Plain-English explanation | When to use it and caveat |
|---|---|---|---|---|
| `reference.ground_truth_shortest_path` | Solver reference route | integer array; required, non-null | One route with minimum total edge weight for the endpoint pair. | Score predictions. Equal-cost alternative routes can also be correct; withhold this field from the model. |
| `reference.gold_cost` | Minimum route cost | integer; required, non-null | Minimum possible sum of edge weights between the endpoints. | Compare the predicted route's cost with the optimum. It is cost, not hop count, and must not be model input. |

#### Provenance fields

| Exact original field path | Human-readable display name | JSON type and optionality | Plain-English explanation | When to use it and caveat |
|---|---|---|---|---|
| `provenance.num_nodes` | Node count | integer; required, non-null | Number of graph nodes. | Describe graph size. Not a performance measure. |
| `provenance.num_edges` | Edge count | integer; required, non-null | Number of distinct undirected edges. | Describe graph size/density. An edge is counted once although listed at both endpoints. |
| `provenance.topology_type` | Graph family | string; required, non-null | Generator-family label, such as `fat_tree`, `grid`, or `barabasi_albert`. | Group graph families. A category, not a performance score. |
| `provenance.topology_dimension` | Graph-family settings | object; required, non-null | Family-specific construction parameters. | Interpret the graph configuration. Keys vary with family and are not comparable as a universal difficulty scale. |
| `provenance.topology_dimension.n` | Node-count or grid-side setting | integer; optional by graph family | For Barabási–Albert, Erdős–Rényi, and jellyfish records, `n` is total node count; for a square grid it is side length, giving `n × n` nodes. | Interpret the family settings. Meaning depends on `provenance.topology_type`. |
| `provenance.topology_dimension.m` | Attachment setting | integer; optional by graph family | Barabási–Albert setting for how many edges each new node attaches with. | Interpret that graph generator. Not an observed edge count. |
| `provenance.topology_dimension.p` | Edge-probability setting | number; optional by graph family | Erdős–Rényi probability used for possible edges. | Interpret that graph generator. It is a construction parameter, not a measured graph statistic. |
| `provenance.topology_dimension.h` | Tree-height setting | integer; optional by graph family | Height parameter for the balanced-tree graph used by the fat-tree family. | Interpret that graph generator. Not the task's route hop count. |
| `provenance.topology_dimension.r` | Tree-branching setting | integer; optional by graph family | Branching-factor parameter for the balanced-tree graph used by the fat-tree family. | Interpret that graph generator. Not a task endpoint or route value. |
| `provenance.topology_dimension.max_degree` | Maximum-degree setting | integer; optional by graph family | Maximum degree setting used by the jellyfish generator. | Interpret that graph generator. It is a construction limit, not necessarily every node's degree. |
| `provenance.weight_distribution` | Edge-weight method | string; required, non-null | Recorded method used to assign edge weights. | Describe generation settings. It is not a statistic of the realized edge weights. |
| `provenance.gold_path_hops` | Reference-route hop count | integer; required, non-null | Number of edges in the stored solver reference route. | Describe reference-route length. Hop count is not the shortest-path objective. |
| `provenance.hop_bucket` | Hop-range category | string; required, non-null | Recorded grouping label for the route's hop count. | Inspect source grouping. It is a category, not observed model difficulty or performance. |
| `provenance.selected_exact_hop` | Selected hop category | integer; required, non-null | Hop stratum recorded during endpoint-pair selection. | Audit sampling metadata. Not an instruction to the model. |
| `provenance.min_optimal_hops` | Fewest hops among optimal routes | integer; required, non-null | Minimum edge count among all minimum-cost routes for the endpoint pair. | Describe route structure. Cost, not hops, defines optimality. |
| `provenance.max_optimal_hops` | Most hops among optimal routes | integer; required, non-null | Maximum edge count among all minimum-cost routes. | Describe alternate optimal-route lengths. Not a performance measure. |
| `provenance.min_total_cost` | Duplicate minimum-cost check | integer; required, non-null | Recorded minimum cost, checked to equal `reference.gold_cost`. | Check label agreement. It is not a second threshold. |
| `provenance.feasible_exact_hops` | Available hop categories | integer array; required, non-null | Hop categories with eligible endpoint pairs during selection. | Inspect construction strata. Sampling metadata, not model input. |
| `provenance.eligible_pair_count_total` | Eligible endpoint pairs overall | integer; required, non-null | Number of pairs meeting the selection rules across available categories. | Describe the candidate pool. Not a count of model responses. |
| `provenance.eligible_pair_count_selected_exact_hop` | Eligible pairs in selected category | integer; required, non-null | Number of eligible pairs in the selected hop category. | Describe the selected candidate stratum. Not the number of evaluation records. |
| `provenance.graph_seed` | Graph-generation seed | integer; required, non-null | Seed recorded for generating graph structure. | Trace construction provenance. A seed alone may not reproduce output across software versions. |
| `provenance.weight_seed` | Weight-generation seed | integer; required, non-null | Seed recorded for assigning edge weights. | Trace weight-generation provenance. Not an edge weight. |
| `provenance.pair_seed` | Pair-selection seed | integer; required, non-null | Seed recorded for selecting endpoints. | Trace sampling provenance. Not model input. |
| `provenance.generation_attempt_index` | Accepted attempt index | integer; required, non-null | Zero-based index of the attempt that produced the task. | Audit construction history. Not model effort. |
| `provenance.attempts_to_accept` | Attempts before acceptance | integer; required, non-null | Number of generation attempts recorded before accepting the task. | Audit construction history. Not model attempts or task difficulty. |
| `provenance.replicate` | Replicate identifier | integer; required, non-null | Generator replicate label. | Trace provenance. Not a score. |
| `provenance.slot_id` | Generator slot identifier | string; required, non-null | Slot label associated with generation. | Trace provenance. Opaque label, not a scientific score. |
| `provenance.support_row` | Supporting-row identifier | integer; required, non-null | Row identifier in supporting generation data. | Trace provenance. Not this JSONL's row number. |
| `provenance.pair_orientation_swapped` | Endpoint orientation flag | boolean; required, non-null | Whether the selected pair order was reversed from canonical order. | Understand how ordered endpoints were selected. `model_input.node_pair` remains authoritative. |
| `provenance.graph_structure_sha256` | Unweighted graph fingerprint | string (64-character hexadecimal SHA-256); required, non-null | Identifies graph structure without edge weights. | Compare unweighted graph identity. Fingerprint overlap can reveal shared structures across different cases. |
| `provenance.weighted_graph_sha256` | Weighted graph fingerprint | string (64-character hexadecimal SHA-256); required, non-null | Identifies graph structure with edge weights. | Compare weighted-graph identity. It does not by itself establish case independence. |
| `provenance.case_instance_sha256` | Task-instance fingerprint | string (64-character hexadecimal SHA-256); required, non-null | Identifies the weighted graph and ordered endpoints. | Compare task-instance identity. A hash is an identity aid, not a correctness or rights claim. |

### D. Evaluation Populations and Dataset Overlaps

The four files contain **3,046 view memberships** but **1,859 unique case
IDs**. A row count is a membership count; summing rows across views does not
give the number of distinct tasks.

The source-defined view labels are backed by observed records:

- G08 Balanced has 100 cases each labeled Easy, Medium, and Hard in
  `view.source_difficulty`. G08 TrainDist has 165 Easy, 112 Medium, and 23
  Hard cases. The source G08 records identify these labels as `balanced_300`
  and `train_distribution_300`.
- G09 omits `view.source_difficulty`. If its `provenance.gold_path_hops` is
  grouped using the G06 hop bands (1–3 Easy, 4–5 Medium, 6–8 Hard), G09
  Balanced has 282 cases in each group; G09 TrainDist has 880 Easy, 600
  Medium, and 120 Hard. This is a descriptive grouping of released records,
  not a G09 field or a reconstruction of a sampling algorithm.

The G09 TrainDist counts are exactly eight times the G06 Train-200 quotas
(110/75/15); the G08 TrainDist counts are the corresponding 300-case
proportions rounded to whole cases. This confirms the realized distribution
pattern, but not every step of the upstream sampling procedure.

These counts support the view labels and show their realized composition.
They do not disclose every upstream selection rule. Graph/path strata are
generation or task metadata, not observed model difficulty or performance.

Pairwise intersections by `production_case_id`, computed from the four files:

| Evaluation views | Shared case IDs |
|---|---:|
| G08 Balanced-300 × G08 TrainDist-300 | 0 |
| G08 Balanced-300 × G09 Balanced-846 | 164 |
| G08 Balanced-300 × G09 TrainDist-1600 | 221 |
| G08 TrainDist-300 × G09 Balanced-846 | 110 |
| G08 TrainDist-300 × G09 TrainDist-1600 | 265 |
| G09 Balanced-846 × G09 TrainDist-1600 | 617 |

The G08 views are disjoint and partition their 600 published cases. The G09
views have 617 cases in common; their union is 1,829 unique cases, leaving 127
of the 1,956 G09 pool cases in neither released G09 view. The evaluation-view
union is 1,859 unique cases. All evaluation case IDs are in G00 and none are
in G06 Train-200 or LossVal-44. This is case-ID separation from these specific
G06 splits, not a claim about every external training corpus.

The overlap counts above are by case ID, not by unweighted graph structure.
Fingerprint comparison confirms that different case IDs can reuse the same
unweighted structure: for example, G09 Balanced case `PR01238` and G09
TrainDist case `PR01236` share `graph_structure_sha256` but have different
weighted-graph fingerprints. Do not interpret case-ID disjointness as graph-
level disjointness. G00 is the parent task universe, includes tasks used in
other stages, and is not an independent blind test set.

### E. Correct Evaluation Usage and Limitations

1. Give the model `model_input` only: the weighted graph and ordered endpoints.
2. Keep `reference` hidden during generation; use it afterward to score the
   predicted route and cost.
3. Report the exact view name and count both memberships and unique case IDs
   when combining views. Account for the overlaps above.
4. Treat `view.source_difficulty`, hop counts, hop buckets, graph family,
   and sampling metadata as task/provenance descriptors, not model outcomes.

The released resources contain evaluation tasks and solver reference labels,
not generated model predictions or performance measurements. The reference
answers are intentionally present in the files, so a careless loader can leak
them into model input. The adapter separates `model_input`, `reference`, and
`provenance`, but researchers must preserve that separation in their own
evaluation code.

**Worked example — `PR01238`, G09 Balanced.** This is an actual compact row;
the provenance object below shows selected fields only.

```json
{
  "schema_version": "pathfact-evaluation-v0.1",
  "production_case_id": "PR01238",
  "view": {"source_stage": "G09", "name": "balanced"},
  "model_input": {
    "topology": {
      "0": [[1, 70]],
      "1": [[0, 70], [2, 132]],
      "2": [[1, 132], [3, 113]],
      "3": [[2, 113], [4, 119]],
      "4": [[3, 119], [5, 162]],
      "5": [[4, 162]]
    },
    "node_pair": [5, 0]
  },
  "reference": {
    "ground_truth_shortest_path": [5, 4, 3, 2, 1, 0],
    "gold_cost": 596
  },
  "provenance": {
    "num_nodes": 6,
    "num_edges": 5,
    "topology_type": "fat_tree",
    "gold_path_hops": 5,
    "hop_bucket": "5-8",
    "weighted_graph_sha256": "596f78f224c4fabec712bbb676b84ba678ed0223cf4f4ac5ae437df1d0b65128"
  }
}
```

This excerpt keeps the task input separate from its solver reference. G09 has
no `view.source_difficulty`; its route metadata remains available under
`provenance`.

## Derived graph resources

These contain graph/path/cost evidence, not model responses, learned answer
likelihoods, or StepPref results. Graph objects throughout use `topology`
and `node_pair`.

### `counterfactual/counterfactual_triples.jsonl` (354 rows)

This file contains 354 graph-derived Original/Stay/Switch triples. Each triple
keeps the same nodes, links, and endpoint pair while changing one link's
weight. The cases come from one or both G09 evaluation views.

#### A. What Is a Counterfactual Triple?

A counterfactual triple is one shortest-path task shown under three versions
of its weighted graph. **Original** is the source graph. **Stay** changes the
selected edge's weight but keeps the designated old route optimal. **Switch**
changes that same edge farther in the same direction, so the designated old
route becomes strictly more costly than the new optimum. The graph structure
and endpoint pair remain fixed.

These records provide controlled graph and solver evidence. They do not
contain model answers, preference scores, training adaptations, or observed
model performance.

#### B. Original, Stay and Switch Graphs

- `original_topology` is the source graph; `node_pair` is the ordered source
  and destination. `variants.original.topology` repeats the Original graph.
- `edited_edge` names the one undirected edge whose weight changes. Its
  Original, Stay, and Switch weights are in the three top-level weight fields.
- `edit_direction` is `increase` or `decrease`. In all 354 records, an
  increase has Original < Stay < Switch weights; a decrease has Original >
  Stay > Switch weights. No other edge weight or graph connection changes.
- `designated_old_path` is the source reference route, which remains optimal
  in Original and Stay. `designated_new_path` is the Switch reference route,
  which is optimal in Switch.
- For increases, the designated old route uses the edited edge and the new
  route avoids it. For decreases, the new route uses the edited edge and the
  old route avoids it. This pattern is verified across the released triples.

The edit is selected so Stay does not yet change which route is cheapest;
Switch does. Specifically, the old route's Stay cost equals the Stay solver
minimum. In Switch, its cost is strictly greater than the solver minimum, and
the new route's cost equals that minimum.

#### C. Field-by-Field Data Dictionary

All top-level and nested fields below are present and non-null in every
released record. Arrays of node IDs are integer paths; graph topologies are
adjacency objects whose node-ID keys are dynamic. Each `[neighbor, weight]`
pair describes an undirected link and its positive integer weight.

| Exact original field path | Plain-English display name | JSON type and nullability | Clear explanation | Research use and interpretation caveat |
|---|---|---|---|---|
| `production_case_id` | Case ID | string; non-null | Identifies the source task and its triple. | Join to the G09 parent task. It identifies a case, not a separate case for every source-view membership. |
| `source_views` | G09 source-view memberships | array of strings; non-null | Lists `G09_Balanced`, `G09_TrainDist`, or both for the source task. | Track where a case was included. Two entries mean one case belongs to two views, not two independent tasks. |
| `source_case_instance_sha256` | Source-task fingerprint | 64-character hexadecimal string; non-null | Fingerprint of the source weighted graph and ordered endpoints. | Check source-task identity against its G09 parent. A hash is an identity aid, not a correctness or rights claim. |
| `source_artifacts` | Source-artifact fingerprints | object; non-null | Groups hashes for the source counterfactual input, construction script, and G09 parent files. | Track provenance and identify the frozen inputs/code. Hashes do not themselves prove scientific validity. |
| `source_artifacts.evaluation_input_sha256` | Evaluation-input fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the frozen counterfactual evaluation input artifact. | Identify the input used to construct the triples. It is not a hash of this output record. |
| `source_artifacts.construction_script_sha256` | Construction-script fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the script recorded as constructing the counterfactual input. | Identify construction provenance. The hash does not describe the algorithm in human-readable form. |
| `source_artifacts.g09_balanced_sha256` | G09 Balanced source fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the recorded G09 Balanced source artifact. | Check source provenance. Not a case-level graph fingerprint. |
| `source_artifacts.g09_traindist_sha256` | G09 TrainDist source fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the recorded G09 TrainDist source artifact. | Check source provenance. Not a case-level graph fingerprint. |
| `node_pair` | Start and destination | array of 2 integers; non-null | Ordered source and destination nodes shared by all three variants. | Interpret every path and solver cost using these endpoints. Preserve the order. |
| `original_topology` | Original weighted graph | object; non-null | Adjacency map for the source graph before the edge edit. | Inspect or reconstruct the Original task. Its duplicate is `variants.original.topology`. |
| `variants` | Three graph variants | object; non-null | Contains `original`, `stay`, and `switch` graph objects and their solver costs. | Compare the same task under three edge weights. The variants are not three independently sampled graphs. |
| `variants.original` | Original variant | object; non-null | The unedited source graph and its minimum route cost. | Establish the baseline. The topology duplicates `original_topology`. |
| `variants.original.topology` | Original variant graph | object; non-null | Original adjacency map. | Use as the baseline graph. It must equal `original_topology`. |
| `variants.original.gold_cost` | Original solver minimum | integer; non-null | Minimum route cost between `node_pair` in the Original graph. | Baseline for checking routes. It is an algorithmically verified graph label, not a model prediction. |
| `variants.stay` | Stay variant | object; non-null | Graph after the Stay edge-weight change and its solver minimum. | Check that the old designated route remains optimal after a smaller/less decisive change. |
| `variants.stay.topology` | Stay variant graph | object; non-null | Adjacency map after changing only `edited_edge`. | Compare with Original. Nodes and links remain the same; only that edge's weight changes. |
| `variants.stay.gold_cost` | Stay solver minimum | integer; non-null | Minimum route cost in the Stay graph. | Compare with `path_costs.stay_old`; equality verifies the old route remains optimal. |
| `variants.switch` | Switch variant | object; non-null | Graph after the Switch edge-weight change and its solver minimum. | Check the change in the optimal route. It is the same task structure, not a new sampled graph. |
| `variants.switch.topology` | Switch variant graph | object; non-null | Adjacency map after changing only `edited_edge` to its Switch weight. | Compare route costs under the changed edge weight. Other weights and links remain fixed. |
| `variants.switch.gold_cost` | Switch solver minimum | integer; non-null | Minimum route cost in the Switch graph. | Compare designated route costs to the new minimum. This is not necessarily the edited-edge weight. |
| `edited_edge` | Changed undirected edge | array of 2 integers; non-null | The two node IDs joined by the edge whose weight changes. | Find the single edited link in each topology. Its order is not a direction of travel. |
| `edit_direction` | Weight-change direction | string; non-null | `increase` or `decrease` relative to the Original edge weight. | Interpret the ordering of the three recorded weights. It does not by itself say which route will be optimal. |
| `original_edge_weight` | Original edge weight | integer; non-null | Weight of `edited_edge` in Original. | Baseline for the two changes. |
| `stay_edge_weight` | Stay edge weight | integer; non-null | Weight of `edited_edge` in Stay. | Check the Stay perturbation. The old route remains solver-optimal at this value. |
| `switch_edge_weight` | Switch edge weight | integer; non-null | Weight of `edited_edge` in Switch. | Check the Switch perturbation. Its route effect is established by the solver/path fields, not by this number alone. |
| `designated_old_path` | Old designated route | integer array; non-null | Source route retained as optimal in Original and Stay. | Track the route whose optimality is tested across variants. It becomes strictly suboptimal in Switch. |
| `designated_new_path` | New designated route | integer array; non-null | Solver reference route selected for Switch. | Track the route optimal in Switch. Its cost in Original need not be minimum. |
| `reference_paths` | Solver reference routes | object; non-null | Stores one solver reference route for each of Original, Stay, and Switch. | Inspect a concrete optimum in each graph. It does not list every route tied for minimum cost. |
| `reference_paths.original` | Original reference route | integer array; non-null | One solver-optimal route in Original. | Check the baseline answer; it equals the designated old path in this release. |
| `reference_paths.stay` | Stay reference route | integer array; non-null | One solver-optimal route in Stay. | Check the Stay answer. Other routes, including the old path, may tie at the minimum. |
| `reference_paths.switch` | Switch reference route | integer array; non-null | One solver-optimal route in Switch. | Check the new optimum; it equals the designated new path in this release. |
| `solver_costs` | Solver minimum costs | object; non-null | Groups the true shortest-path cost for each variant. | Compare route costs with the minimum for that same graph. These values come from graph solving, not a model. |
| `solver_costs.original` | Original minimum cost | integer; non-null | Minimum path cost in Original. | Baseline optimum. |
| `solver_costs.stay` | Stay minimum cost | integer; non-null | Minimum path cost in Stay. | Verify the old route's continued optimality. |
| `solver_costs.switch` | Switch minimum cost | integer; non-null | Minimum path cost in Switch. | Verify which route wins after the switch. |
| `path_costs` | Designated-route costs | object; non-null | Groups costs of the old/new designated paths in selected graph variants. | Compare specific routes across edits. These are not all shortest-path costs. |
| `path_costs.original_old` | Old route cost in Original | integer; non-null | Cost of `designated_old_path` in Original. | Compare with `solver_costs.original`; they are equal. |
| `path_costs.original_new` | New route cost in Original | integer; non-null | Cost of `designated_new_path` in Original. | Show the new route's baseline cost; it may be above the Original minimum. |
| `path_costs.stay_old` | Old route cost in Stay | integer; non-null | Cost of `designated_old_path` in Stay. | Compare with `solver_costs.stay`; equality shows it remains optimal. |
| `path_costs.switch_old` | Old route cost in Switch | integer; non-null | Cost of `designated_old_path` in Switch. | Compare with `solver_costs.switch`; it must be strictly greater. |
| `path_costs.switch_new` | New route cost in Switch | integer; non-null | Cost of `designated_new_path` in Switch. | Compare with `solver_costs.switch`; equality shows it is optimal. |
| `optimality_relations` | Verified route relationships | object; non-null | Records four solver-checked claims about designated routes. | Use as concise relation flags; inspect costs for magnitudes. These are graph properties, not model judgments. |
| `optimality_relations.original_old_optimal` | Old route optimal in Original | boolean; non-null; `true` in all rows | The old route's Original cost equals the Original solver minimum. | Confirm the baseline old route is optimal. |
| `optimality_relations.stay_old_optimal` | Old route optimal in Stay | boolean; non-null; `true` in all rows | The old route's Stay cost equals the Stay solver minimum. | Confirm Stay preserves the old optimum. |
| `optimality_relations.switch_old_strictly_suboptimal` | Old route suboptimal in Switch | boolean; non-null; `true` in all rows | The old route costs strictly more than the Switch solver minimum. | Confirm the route ranking changes in Switch. |
| `optimality_relations.switch_new_optimal` | New route optimal in Switch | boolean; non-null; `true` in all rows | The new route's Switch cost equals the Switch solver minimum. | Confirm the Switch route is optimal. |

#### D. Understanding Route Costs and Optimality

`path_costs` answers, “What does this particular designated route cost in
this variant?” `solver_costs` answers, “What is the cheapest possible route
cost in this variant?” The first is route-specific; the second is the optimum
over all routes. Therefore, an old route can cost more than the minimum in
Switch while another route achieves the solver cost.

The four `optimality_relations` are verified from those comparisons:

- `original_old_optimal`: `path_costs.original_old == solver_costs.original`.
- `stay_old_optimal`: `path_costs.stay_old == solver_costs.stay`.
- `switch_old_strictly_suboptimal`: `path_costs.switch_old > solver_costs.switch`.
- `switch_new_optimal`: `path_costs.switch_new == solver_costs.switch`.

**Worked example — `PR00043`.** This released record has
`node_pair=[1,5]`, `edited_edge=[1,5]`, and `edit_direction="increase"`.
The edge weights are Original 160, Stay 169, and Switch 171. Solver minima
are 160, 169, and 170, respectively. The old path is `[1,5]`: it costs 160
in Original, 169 in Stay, and 171 in Switch. In Switch, the record's actual
new path is `[1,0,5]`. Its edges were not edited: `[1,0]` costs 93 and `[0,5]`
costs 77, so this route costs 93 + 77 = 170. It is therefore cheaper than
the 171-cost direct edge and matches the Switch solver minimum. No replacement
route needs to be inferred; it is recorded in `designated_new_path` and
`reference_paths.switch`.

#### E. Provenance and Scientific Limitations

`source_views` records membership in the G09 Balanced and/or G09 TrainDist
source views. The 354 unique case IDs account for 482 memberships: 179 in
G09_Balanced and 303 in G09_TrainDist. There are 128 cases in both views and
226 in one view only. Thus 482 is a count of source-view memberships, not 482
independent graph tasks.

The triples are derived by changing one graph edge and checking route costs
with a shortest-path solver. They are graph-derived controlled comparisons,
not model predictions, learned preferences, or model adaptation/training
results. They contain no generated model responses or performance measures.
The source and artifact hashes support provenance and identity checks; they do
not establish model behavior, universal causal effects, or rights by
themselves.

### `training/ranking_eligibility.jsonl` (200 rows)

This derived resource links to all 200 case IDs in G06 Train-200. It records
whether the old and new candidate routes have a usable graph-cost comparison
on an Original graph and, when available, a Switch graph. The labels describe
route costs in weighted graphs. They are not model outputs or model preference
scores.

#### A. What Is Ranking Eligibility?

The file identifies which G06 training cases support a particular
counterfactual route-comparison setup. The cases originate from G06 Train-200:
the construction code checks that the ranking records, eligibility records,
and released G06 Gold Train-200 records have the same 200 unique
`production_case_id` values. This makes the resource useful for studying the
graph conditions available for training-related ranking objectives. It does
not make these rows independent held-out evaluation tasks.

For each case, the old route is the route used as the original G06 reference.
Where the source supplies a valid Switch graph and a new candidate route, the
record compares both routes on both graphs. Route costs and solver minima
support the eligibility labels; no model score is included.

#### B. Comparing the Old and New Routes

`original_graph` contains the weighted graph and ordered endpoints for the
G06 task. `switch_graph`, when present, has the same nodes, links, and
endpoints, with the selected `edited_edge` assigned `new_edge_weight` instead
of `old_edge_weight`. `old_path` is the G06 reference route. `new_path`, when
present, is the solver-optimal route for the supplied Switch graph.

`path_costs` gives the cost of each named route in each graph. `solver_costs`
gives the minimum possible route cost in each graph. The two are related but
answer different questions: a particular route can cost more than the
minimum. In the 171 supplied Switch comparisons, the old route is optimal in
Original and the new route is optimal in Switch; the categories further
classify how the two candidates compare in Original.

The dataset has three categories:

- **`strict_bidirectional` (162):** the old route costs strictly less than the
  new route in Original, and the new route costs strictly less than the old
  route in Switch. These cases support strict graph-cost preference labels
  in both directions.
- **`original_tie_switch_strict` (9):** both candidate routes have equal cost
  in Original, while the new route is strictly cheaper in Switch. A Switch
  comparison exists, but there is no strict Original preference; these rows
  therefore do not meet the bidirectional criterion.
- **`no_strict_switch` (29):** the G06 source marks no qualifying Switch
  counterfactual as available. Only the Original old route and its solver
  minimum are retained, so these rows cannot supply a two-graph comparison.

Thus 171 rows have Switch graphs and both candidate routes' costs on both
graphs (162 + 9). Only 162 of those show strict route preference in both
directions. For the other 29, switch-related values are null because the
source did not provide an eligible Switch comparison; null does not mean zero
cost or a failed solver run.

#### C. Understanding the Three Eligibility Categories

The category is derived from graph route costs:

- `strict_bidirectional` requires
  `original_old < original_new` and `switch_new < switch_old`.
- `original_tie_switch_strict` requires
  `original_old == original_new` and `switch_new < switch_old`.
- `no_strict_switch` is assigned when the source eligibility mode is
  `SFT_ONLY`; no Switch comparison is emitted.

The source uses “preference” to name these cost orderings. They are labels
derived from the graph, not human judgments, model choices, answer-token
negative log-likelihood (NLL) margins, or proof that a ranking loss is
satisfied. A model's token-level scores and training behavior would require
separate measurements.

#### D. Field-by-Field Data Dictionary

All top-level keys are present in every record. `null` values and nested
fields available only when `switch_graph` is an object are identified below.
Graph topology objects use stringified node IDs as dynamic keys; each value
is a list of `[neighbor_id, positive_integer_weight]` pairs for an undirected
graph. Node IDs inside paths and endpoint arrays are JSON integers.

| Exact original field path | Human-readable display name | JSON type and nullability | Plain-English explanation | Intended research use and interpretation caveat |
|---|---|---|---|---|
| `production_case_id` | G06 case ID | string; non-null | Identifies the parent G06 task. | Join this row to G06 Train-200. It is not a unique evaluation-case identifier independent of G06. |
| `eligibility_category` | Route-comparison category | string; non-null | One of `strict_bidirectional`, `original_tie_switch_strict`, or `no_strict_switch`. | Select cases by the available graph-cost relation. The label describes construction eligibility, not model performance. |
| `counterfactual_available` | Switch comparison available | boolean; non-null | `true` when the source supplied a Switch comparison; `false` for the 29 `no_strict_switch` rows. | Filter rows with two-graph comparisons. `true` does not guarantee strict preference in Original; 9 such rows tie there. |
| `original_graph` | Original task graph and endpoints | object; non-null | The G06 weighted graph before the edge change, together with the task's endpoint pair. | Reconstruct the original shortest-path task. Its route costs are graph-derived labels, not model scores. |
| `original_graph.topology` | Original weighted topology | object; non-null | Adjacency map for the Original graph, keyed by node ID. | Recompute route costs or shortest paths. Dynamic node-ID keys are graph contents, not additional schema fields. |
| `original_graph.node_pair` | Original start and destination | array of 2 integers; non-null | Ordered source and destination nodes for the task. | Interpret both paths and graph costs against these endpoints; do not reorder them. |
| `switch_graph` | Switch graph and endpoints | object or null; null in 29 rows | The counterfactual graph after changing one edge weight, with the same endpoint pair. | Compare route costs after the edit when present. Null means no Switch comparison was supplied by the source. |
| `switch_graph.topology` | Switch weighted topology | object; non-null when `switch_graph` is an object | Adjacency map after changing the selected edge weight. | Compare against Original; construction preserves the nodes and links and changes the selected edge's weight. |
| `switch_graph.node_pair` | Switch start and destination | array of 2 integers; non-null when `switch_graph` is an object | The ordered endpoints for the Switch task. | Confirm the same task is compared; it matches `original_graph.node_pair`. |
| `edited_edge` | Changed edge | array of 2 integers or null; null in 29 rows | The two node IDs connected by the edge whose weight changes. | Identify the changed undirected link. The pair's order is not travel direction. |
| `old_edge_weight` | Edge weight before Switch | integer or null; null in 29 rows | Weight of `edited_edge` in Original. | Compare with `new_edge_weight`; absent when no Switch comparison is supplied. |
| `new_edge_weight` | Edge weight in Switch | integer or null; null in 29 rows | Replacement weight assigned to `edited_edge` in Switch. | Check the graph edit. It alone does not determine which full route is cheapest. |
| `old_path` | Old/reference route | array of integers; non-null | G06's reference route for the Original task. | Compare its cost across Original and Switch. It is solver-optimal in Original, not necessarily in Switch. |
| `new_path` | Switch-optimal route | array of integers or null; null in 29 rows | Solver-optimal candidate route supplied with the Switch comparison. | Compare against `old_path` in both graphs. It is absent when no Switch comparison is available. |
| `path_costs` | Candidate-route costs | object; non-null | Costs of the named old and new routes on the graphs where the comparison exists. | Compare candidates directly. These are specific route costs, not necessarily graph minima. |
| `path_costs.original_old` | Old route cost in Original | integer; non-null | Cost of `old_path` on `original_graph`. | Compare with `solver_costs.original`; equal in all rows. |
| `path_costs.original_new` | New route cost in Original | integer or null; null in 29 rows | Cost of `new_path` on `original_graph`. | Distinguishes a strict Original preference from a tie when Switch data exists. |
| `path_costs.switch_old` | Old route cost in Switch | integer or null; null in 29 rows | Cost of `old_path` on `switch_graph`. | Compare with the new route and Switch minimum when present. |
| `path_costs.switch_new` | New route cost in Switch | integer or null; null in 29 rows | Cost of `new_path` on `switch_graph`. | Compare with `solver_costs.switch`; equal in supplied Switch comparisons. |
| `solver_costs` | Graph minimum route costs | object; non-null | Groups solver-computed shortest-path costs for Original and, when available, Switch. | Check candidate optimality. A minimum is not a model answer or loss value. |
| `solver_costs.original` | Original minimum cost | integer; non-null | Minimum route cost between the Original endpoints. | Compare with Original path costs; it equals `path_costs.original_old`. |
| `solver_costs.switch` | Switch minimum cost | integer or null; null in 29 rows | Minimum route cost in Switch when that graph is present. | Validate the Switch-optimal route; null when there is no supplied Switch comparison. |
| `preference_relations` | Graph-derived route relations | object; non-null | Records the cost-based relation between old and new routes in each graph. | Use as a compact categorical summary, then consult `path_costs` for magnitudes. These are not model preference scores. |
| `preference_relations.original` | Original route relation | string; non-null | `old_strictly_preferred`, `tie`, or `not_ranked`. | In this file, the first two occur when Switch data exists; `not_ranked` marks the 29 no-Switch rows. “Preferred” means lower graph cost. |
| `preference_relations.switch` | Switch route relation | string; non-null | `new_strictly_preferred` when Switch data exists, otherwise `not_available`. | `not_available` marks absent comparison data; it is not a tie or model failure. |
| `source_artifacts` | Source fingerprint group | object; non-null | Groups SHA-256 fingerprints for ranking input, eligibility input, G06 Train-200, and construction code. | Trace which frozen artifacts the record was derived from. Fingerprints identify bytes; they do not establish scientific validity. |
| `source_artifacts.ranking_sha256` | Ranking-input fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the archived G06 ranking input. | Identify the frozen ranking source used in construction; not a hash of this output row. |
| `source_artifacts.eligibility_sha256` | Eligibility-input fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the archived eligibility and mode input. | Identify the source of availability/category decisions; not a model score. |
| `source_artifacts.core_g06_train_sha256` | G06 Train-200 fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the released G06 Gold Train-200 parent file. | Trace the parent training split. The file is training-related, not held-out evaluation data. |
| `source_artifacts.construction_script_sha256` | Construction-code fingerprint | 64-character hexadecimal string; non-null | SHA-256 recorded for the script associated with constructing the ranking input. | Identify the recorded construction artifact; the hash itself does not explain or validate its algorithm. |

#### E. Research Uses and Scientific Limitations

Researchers can use this resource to identify which G06 Train-200 cases have
strict bidirectional graph-cost comparisons, Original ties followed by a
strict Switch comparison, or no supplied Switch comparison. It can support
analysis of candidate selection or preparation of training-related
counterfactual ranking data. These rows share case IDs with G06 Train-200;
they are not independent held-out examples.

The graph-cost gaps (for example, 244 versus 207) are not answer-token NLL
margins. They do not show that a model prefers either route, that a ranking
loss would be satisfied, or that training would improve performance. Such
claims require separately specified model inputs, scoring or training
procedures, and empirical results. The released records contain graph and
solver-derived comparisons, not model predictions or training outcomes.

### `training/boundary_local.jsonl` (162 rows)

This resource contains local Stay/Switch graph variants for 162 G06
Train-200 tasks. It moves the two candidate routes to either side of a
calculated edge-weight at which their costs tie. The construction then checks
that the designated winning route is also a shortest path in each local
graph.

#### A. What Is Boundary-Local?

Each row is linked to a case in the `strict_bidirectional` subset of G06
Train-200. Construction requires the Boundary-Local case IDs to match that
162-case subset exactly. In the Original graph, `old_path` is strictly
cheaper than `new_path`; these are the same parent graph and paths recorded
for the G06 ranking case.

The resource makes two nearby versions of that same graph. In Local Stay,
the old route remains strictly cheaper. In Local Switch, the new route
becomes strictly cheaper. The rows provide graph and solver-derived
comparisons for research on training-related ranking setups; they contain no
model preference measurements or autonomous pathfinding results.

#### B. Understanding the Route-Cost Tie Boundary

The `boundary_tie_weight` is the edited-edge weight at which the recorded
`old_path` and `new_path` would have equal total route cost, holding every
other edge fixed. Let each route's base cost be its cost excluding the edited
edge, and let its edge-use indicator be 1 if it uses that edge and 0 if it
does not. The tie solves
`old_base + old_uses_edge × weight = new_base + new_uses_edge × weight`,
so `boundary_tie_weight = (new_base - old_base) / (old_uses_edge -
new_uses_edge)`. The two routes use the edge differently in every row, so the
denominator is 1 or -1; the builder also requires a positive integer tie
weight.

The local weights are set one integer unit on either side of that boundary.
The side assigned to Stay or Switch depends on which route uses the edge, but
the resulting route-cost gaps are fixed: in Local Stay, new costs exactly 1
more than old; in Local Switch, old costs exactly 1 more than new. The
solver checks also require the designated Stay winner (old) and Switch
winner (new) to be globally shortest, so a third route cannot undercut them
(it may tie the minimum).

For example, when the old route avoids the edited edge and the new route
uses it, lowering the edge weight favors the new route. In that case the
tie weight is `boundary_tie_weight`; Stay is one unit above it and Switch is
one unit below it. When the routes use the edge in the opposite pattern, the
weight direction is reversed while the same route-cost conditions hold.

#### C. Original, Local Stay and Local Switch

- `original_graph` is the G06 parent task graph. Its endpoints, topology,
  old/new paths, edited edge, and Original edge weight are checked against
  the strict-bidirectional ranking record.
- `local_stay_graph` keeps the same nodes, links, and endpoints, changing
  only the edited edge to `stay_edge_weight`. The old route remains strictly
  cheaper than the new route and is a solver-confirmed global optimum.
- `local_switch_graph` changes only that edge to `switch_edge_weight`. The
  new route becomes strictly cheaper and is a solver-confirmed global
  optimum.

In every row, the three topologies preserve the same edge set and all
unmodified edge weights. The `old_path` and `new_path` are the two designated
candidate routes carried from the G06 parent. Which one traverses the edited
edge varies by case. The recorded edge-use flags show that exactly one of
the two candidate routes uses it.

#### D. Field-by-Field Data Dictionary

All top-level fields and nested fields below are present and non-null in all
162 records. Graph topologies are adjacency objects keyed by stringified
node IDs; each node's value is a list of `[neighbor_id, positive_integer_weight]`
pairs describing an undirected graph. The graph node keys are data, not
additional fixed schema fields. Paths and endpoint arrays contain JSON
integers.

| Exact original field path | Human-readable display name | JSON type and nullability | Plain-English explanation | Intended research use and interpretation caveat |
|---|---|---|---|---|
| `production_case_id` | G06 case ID | string; non-null | Identifies the parent shortest-path task. | Join to G06 Train-200 and its ranking-eligibility row. It is training-related, not a held-out evaluation ID. |
| `eligibility_category` | Parent eligibility category | string; non-null; always `strict_bidirectional` | Marks this row as part of the strict-bidirectional G06 subset. | Confirm the subset relationship. It is a graph-cost category, not a model result. |
| `original_graph` | Original task graph and endpoints | object; non-null | Parent G06 graph before the local boundary edit, with its ordered endpoints. | Reconstruct the original task and compare with the G06 parent. |
| `original_graph.topology` | Original weighted topology | object; non-null | Adjacency map for the Original graph. | Recompute route costs or shortest paths. Dynamic node-ID keys are graph contents. |
| `original_graph.node_pair` | Original start and destination | array of 2 integers; non-null | Ordered source and destination nodes. | Use these endpoints for all three variants; preserve their order. |
| `local_stay_graph` | Local Stay graph and endpoints | object; non-null | Same task after assigning the edited edge its Stay weight. | Study the graph just on the side where the old route wins; it is not a separately sampled task. |
| `local_stay_graph.topology` | Local Stay topology | object; non-null | Adjacency map after changing only the edited-edge weight. | Compare with Original; nodes and links remain fixed. |
| `local_stay_graph.node_pair` | Local Stay start and destination | array of 2 integers; non-null | Ordered endpoints for Local Stay. | Confirm the same task; it matches `original_graph.node_pair`. |
| `local_switch_graph` | Local Switch graph and endpoints | object; non-null | Same task after assigning the edited edge its Switch weight. | Study the side where the new route wins; it is not a new task. |
| `local_switch_graph.topology` | Local Switch topology | object; non-null | Adjacency map after changing only the edited-edge weight. | Compare costs on the other side of the boundary; all other weights remain fixed. |
| `local_switch_graph.node_pair` | Local Switch start and destination | array of 2 integers; non-null | Ordered endpoints for Local Switch. | Confirm the same task; it matches the Original pair. |
| `edited_edge` | Boundary-adjusted edge | array of 2 integers; non-null | Node IDs of the single undirected edge whose weight changes across variants. | Locate the edited edge. Array order does not indicate travel direction. |
| `original_edge_weight` | Original edge weight | integer; non-null | Weight on `edited_edge` in `original_graph`. | Check parent identity and baseline. |
| `stay_edge_weight` | Local Stay edge weight | integer; non-null | Weight assigned to the edited edge in Local Stay. | Check its location relative to the tie boundary and its effect on both routes. |
| `switch_edge_weight` | Local Switch edge weight | integer; non-null | Weight assigned to the edited edge in Local Switch. | Check the opposite-side perturbation. The direction of weight change depends on route edge use. |
| `boundary_tie_weight` | Route-cost tie weight | integer; non-null | Edited-edge weight at which the two designated routes' total costs are equal, with other edges unchanged. | Interpret the local perturbation. It is a calculated tie for these two routes, not a model threshold. |
| `original_weight_vocabulary` | Distinct Original edge weights | array of integers; non-null | Sorted distinct set of edge weights appearing in the Original graph. | Check whether local weights reuse values from the Original graph. Most local weights are outside this set; membership is recorded, not required. |
| `old_path` | Old candidate route | array of integers; non-null | G06 reference route, strictly cheaper in Original and Local Stay. | Compare its cost across the three graphs. It is not globally optimal in Local Switch. |
| `new_path` | New candidate route | array of integers; non-null | G06 counterfactual route, strictly cheaper in Original's parent Switch and Local Switch. | Compare the candidate against `old_path`. It is not globally optimal in Local Stay. |
| `path_costs` | Candidate-route costs | object; non-null | Costs of both named routes in all three variants. | Measure pairwise route gaps. These values are not themselves all-possible-path minima. |
| `path_costs.original_old` | Old route cost in Original | integer; non-null | Total cost of `old_path` in `original_graph`. | Compare with Original solver minimum and the new route. |
| `path_costs.original_new` | New route cost in Original | integer; non-null | Total cost of `new_path` in `original_graph`. | Verify the old route is strictly cheaper in Original. |
| `path_costs.stay_old` | Old route cost in Local Stay | integer; non-null | Total cost of `old_path` in Local Stay. | Compare with `stay_new`; old is exactly 1 cheaper. |
| `path_costs.stay_new` | New route cost in Local Stay | integer; non-null | Total cost of `new_path` in Local Stay. | Pairwise cost is exactly 1 above old; it may tie a third route at the global minimum. |
| `path_costs.switch_old` | Old route cost in Local Switch | integer; non-null | Total cost of `old_path` in Local Switch. | Compare with `switch_new`; old is exactly 1 more costly. |
| `path_costs.switch_new` | New route cost in Local Switch | integer; non-null | Total cost of `new_path` in Local Switch. | It equals the solver minimum; a third route may tie but cannot be cheaper. |
| `solver_costs` | Shortest-path costs by graph | object; non-null | Groups minimum possible route costs for Original, Local Stay, and Local Switch. | Check whether designated winners are globally optimal. A solver cost is not a model score. |
| `solver_costs.original` | Original minimum cost | integer; non-null | Cheapest possible route cost in the Original graph. | Verified equal to `path_costs.original_old`. |
| `solver_costs.stay` | Local Stay minimum cost | integer; non-null | Cheapest possible route cost in Local Stay. | Verified equal to `path_costs.stay_old`; another route may tie. |
| `solver_costs.switch` | Local Switch minimum cost | integer; non-null | Cheapest possible route cost in Local Switch. | Verified equal to `path_costs.switch_new`; another route may tie. |
| `optimality_relations` | Strict route-order checks | object; non-null | Stores three booleans for the old/new cost ordering in Original, Stay, and Switch. | Use as compact checks, then inspect costs for the actual gaps. These are graph facts, not model judgments. |
| `optimality_relations.original_old_strictly_better` | Old strictly cheaper in Original | boolean; non-null; always `true` | Old route's Original cost is less than new route's. | Confirms the strict starting preference used for this subset. |
| `optimality_relations.stay_old_strictly_better` | Old strictly cheaper in Stay | boolean; non-null; always `true` | Old route's Local Stay cost is less than new route's by exactly 1. | Confirms the Stay side of the local boundary; the solver separately checks global optimality. |
| `optimality_relations.switch_new_strictly_better` | New strictly cheaper in Switch | boolean; non-null; always `true` | New route's Local Switch cost is less than old route's by exactly 1. | Confirms the Switch side; the solver separately checks global optimality. |
| `boundary_constraints` | Boundary and edge-use checks | object; non-null | Groups booleans for vocabulary membership, opposite sides of the tie, and candidate-route use of the edited edge. | Audit how the record was constructed. Flags report checks; they do not imply every vocabulary flag is true. |
| `boundary_constraints.both_local_weights_in_original_vocabulary` | Both local weights are Original values | boolean; non-null | True only when both Stay and Switch weights occur among Original graph edge weights. | False in 159 rows and true in 3; using only Original vocabulary values is not required. |
| `boundary_constraints.stay_weight_in_original_vocabulary` | Stay weight occurs in Original | boolean; non-null | Whether `stay_edge_weight` is a member of `original_weight_vocabulary`. | This is descriptive membership, not a pass/fail condition; true in 14 rows. |
| `boundary_constraints.switch_weight_in_original_vocabulary` | Switch weight occurs in Original | boolean; non-null | Whether `switch_edge_weight` is a member of `original_weight_vocabulary`. | This is descriptive membership, not a pass/fail condition; true in 11 rows. |
| `boundary_constraints.stay_switch_are_opposite_sides_of_tie` | Local weights straddle the tie | boolean; non-null; always `true` | Stay and Switch weights are one integer unit on opposite sides of `boundary_tie_weight`. | Confirms the local construction; it does not describe an LLM score margin. |
| `boundary_constraints.old_path_uses_edited_edge` | Old route uses edited edge | boolean; non-null | True when consecutive nodes in `old_path` include `edited_edge`, regardless of edge orientation. | Interpret which route's cost changes with the edge. True in 110 rows, false in 52. |
| `boundary_constraints.new_path_uses_edited_edge` | New route uses edited edge | boolean; non-null | True when consecutive nodes in `new_path` include `edited_edge`, regardless of edge orientation. | Interpret the other route's cost response. True in 52 rows, false in 110; exactly one candidate uses the edge in every row. |
| `source_artifacts` | Provenance fingerprints | object; non-null | Groups hashes for the archived boundary records, protocol, boundary builder, and canonical training script. | Trace the recorded inputs and code. Hashes identify bytes; they do not prove model outcomes or scientific validity. |
| `source_artifacts.boundary_sha256` | Boundary-input fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the archived Boundary-Local source records. | Identify the frozen source artifact; not a hash of this individual output row. |
| `source_artifacts.protocol_sha256` | Protocol fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the recorded Boundary-Local protocol file. | Identify the construction/training protocol version. |
| `source_artifacts.construction_script_sha256` | Boundary-builder fingerprint | 64-character hexadecimal string; non-null | SHA-256 recorded for the script that builds local boundary examples. | Identify construction provenance; a hash does not explain the algorithm. |
| `source_artifacts.canonical_training_script_sha256` | Canonical training-script fingerprint | 64-character hexadecimal string; non-null | SHA-256 of the canonical G06 bidirectional training script recorded in the protocol. | Trace the related training artifact; it is not evidence that this resource contains trained weights or measured performance. |

#### E. Research Uses and Scientific Limitations

Boundary-Local records can support analysis of graph-cost comparisons close
to a two-route tie, or preparation of a controlled local ranking objective.
Their relation to G06 Train-200 means they are training-related resources,
not independent held-out tasks.

Here, “preferred” means lower summed edge weight, and the tie boundary is
for the two recorded candidate routes. The one-unit gaps are differences in
sums of graph-edge weights. They are not answer-token negative log-likelihood
(NLL) margins and do not demonstrate that a model ranks the routes correctly,
that a ranking loss is satisfied, or that training improves ranking or
autonomous pathfinding accuracy. The
released file contains graph variants, candidate routes, solver labels, and
construction metadata—not model predictions or trained-model results.

The protocol notes that most local edge weights are not present in the
Original graph's weight vocabulary. Researchers should account for that
input-distribution change when interpreting an experiment; the membership
flags make it visible but do not remove the limitation.


## Validation and interpretation limits

Hash fields are fingerprints, not semantic guarantees or licensing evidence.
`teacher_vsp_valid` certifies the recorded final path verification, not all
Teacher prose. Optionality in this dictionary means a field key is omitted
from some rows, while nullability means it is present with JSON `null`;
these distinctions are observed in G09 `view.source_difficulty` (omitted)
and in ranking eligibility (nulls). No field meanings should be inferred
solely from a key name where the code/provenance does not define them.
