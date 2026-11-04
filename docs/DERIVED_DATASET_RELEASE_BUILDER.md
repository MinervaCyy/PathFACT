# Derived Dataset Release Builder — Draft v0.1

Status: implemented as a local candidate builder for new derived artifacts.
Its local candidate output is not itself a public release. The operation does not train models, run inference,
reconstruct examples, or include raw model responses.

## Command

`pathfact dataset export-derived-candidate` requires explicit paths for the
finalized counterfactual input and construction script, both Core G09 parent
views, the G06 ranking input and its construction script, the eligibility
resource, Core G06 Train-200, Boundary-Local records, protocol and construction
script, the October 15 scoring report, and a new output directory. It also
requires `--dataset-version`. Run `pathfact dataset export-derived-candidate
--help` for the exact argument names. The output path must not exist.

The builder checks the archived source fingerprints, canonical counts and
case relationships, validates graph/path/cost semantics in memory, creates a
private local candidate, verifies its Manifest v0.1, and checks that source
hashes did not change during the operation. The output is marked
`release_status: draft_not_approved` and `license: null`. This is not a public
release, rights review, or approval.

## Artifacts and output allowlists

Records are deterministically ordered by parent case ID and serialized as
UTF-8 JSONL with sorted JSON keys. No generic source-field copying is used.

### `counterfactual/counterfactual_triples.jsonl`

Exactly 354 records represent graph triples, not view memberships. The
allowlisted top-level fields are:
`production_case_id`, `source_views`, `source_case_instance_sha256`,
`source_artifacts`, `node_pair`, `original_topology`, `variants`,
`edited_edge`, `edit_direction`, `original_edge_weight`, `stay_edge_weight`,
`switch_edge_weight`, `designated_old_path`, `designated_new_path`,
`reference_paths`, `solver_costs`, `path_costs`, and
`optimality_relations`. The nested records contain only the normalized graph,
solver cost, path, condition, and source fingerprint fields needed to verify
the archived task. Prompt text and model responses are omitted. Source view
membership is retained on each unique parent; the verified resource has 482
memberships over 354 cases.

### `training/ranking_eligibility.jsonl`

Exactly 200 G06 parent cases, partitioned as 162 strict bidirectional, 9
Original-tie/Switch-strict, and 29 without a qualifying strict Switch. The
allowlisted fields are `production_case_id`, `eligibility_category`,
`counterfactual_available`, `original_graph`, `switch_graph`, `edited_edge`,
`old_edge_weight`, `new_edge_weight`, `old_path`, `new_path`, `path_costs`,
`solver_costs`, `preference_relations`, and `source_artifacts`. Ineligible
cases have no Switch graph or New candidate. Graph-cost preferences are not
model answer-token NLL margins.

### `training/boundary_local.jsonl`

Exactly 162 cases, each linked to the strict-bidirectional G06 subset. The
allowlisted fields are `production_case_id`, `eligibility_category`,
`original_graph`, `local_stay_graph`, `local_switch_graph`, `edited_edge`,
`original_edge_weight`, `stay_edge_weight`, `switch_edge_weight`,
`boundary_tie_weight`, `original_weight_vocabulary`, `old_path`, `new_path`,
`path_costs`, `solver_costs`, `optimality_relations`,
`boundary_constraints`, and `source_artifacts`. The builder checks the
archived protocol's strict-162 coverage and one-unit objective margin, the
opposite sides of the tie weight, the recorded vocabulary and edge-use flags,
the paths and solver references, and that no unrelated edge changes. It does
not create or claim Boundary-Local task-only VSP results.

## Provenance and exclusions

The manifest records schema version, source stage, data and construction-script fingerprints,
exported hashes, byte sizes and record counts. `metadata/summary.json` records
the count partition, view memberships, the October 15 scoring condition count
and response total, source/report fingerprints, and manifest verification
status. Condition names, checkpoint metadata, raw inference rows, and scoring
response content are not exported. The scoring report is only checked against
the archived input hash, eight conditions, and 1,062 responses per condition.

StepPref, G09 held-out preference results, counterfactual raw outputs, model
completions, prompts, private paths, checkpoint metadata, and experimental
logs are excluded. Historical findings remain attached to their source
protocol and do not certify all Teacher reasoning. File-level content, rights,
third-party terms, attribution, and publication review remain necessary.

Portable synthetic adapter tests exercise graph/path validation, ranking
classification, duplicate rejection, and invalid cost/edit rejection. The
builder has been exercised in memory against the locally archived records;
this implementation task does not create a real release candidate.
