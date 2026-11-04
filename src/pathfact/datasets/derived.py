"""Allowlisted exporters for frozen counterfactual and ranking resources.

These adapters only normalize archived, solver-verified records. They do not
generate examples, run models, or make release-approval claims.
"""

import ast
import hashlib
import heapq
import json
import re
import shutil
import tempfile
import os
from collections import defaultdict
from pathlib import Path

from pathfact.datasets.io import DatasetFormatError, iter_jsonl
from pathfact.datasets.manifest import (
    create_release_manifest,
    verify_release_manifest,
)


CF_SCHEMA = "pathfact-counterfactual-triples-v0.1"
RANK_SCHEMA = "pathfact-ranking-eligibility-v0.1"
BOUNDARY_SCHEMA = "pathfact-boundary-local-v0.1"
CF_SHA256 = "f5879593a53f38943e8b3b991e1ee704c51c58ddd2a3379d524996d62053fb13"
CF_SCRIPT_SHA256 = "35d6f71ebfcee55058c5411b0d56ed377c864df296f738348f79cb16c37fcd5d"
RANK_SCRIPT_SHA256 = "63edd57749c604171e55718a28afb3c1dedbe84197d122bfb36638c5a7cd6014"
BOUNDARY_BUILDER_SHA256 = "d64e62098e204cfe26e3a36e7f9d3254a53b5a7a41d1bb01483bf2983703d83e"
G09_BALANCED_SHA256 = "7dbf359ff55c0f3a246be12ec53b8671d94cde887e88fa831115c5bd456dfc92"
G09_TRAINDIST_SHA256 = "d15f0c21979cbd536660431b2411ad031421c385e73ad2ac438b732f34320a40"
G09_BALANCED_EXPORT_SHA256 = "ae9dad9de11caadabc3c3db1b99cc29c21c90da9de69c368ac7c58e9734bae3c"
G09_TRAINDIST_EXPORT_SHA256 = "a68d3bcb142465bda419cd8edddd37135a049496e186f5feca5c99e396e93d68"
G06_TRAIN_EXPORT_SHA256 = "38ab4d30680b16dcd321924443f526bcb242f2a74d04d6e1ba180d9772d3dc33"
BOUNDARY_PROTOCOL_SHA256 = "5ec96338343b0ec8154deb096e8266f598937d547e521ac46f32710151782a39"
BOUNDARY_TRAIN_SCRIPT_SHA256 = "ecbe4311dfdf83cf668a37f599857ae0d3e6a5739ff09618d17b3155c8f56ab1"
RANK_SHA256 = "61bee8c992e5dbbd9614bf0774e9fee13326e8b1a3c18c54d8c4e59208fd7af3"
ELIGIBILITY_SHA256 = "3b139cc4749c071829d48d02ee15e2b46e5009b5734167ab4c8662987d6814b5"
BOUNDARY_SHA256 = "14b411027c7e2dcecd4cf83bc8edaf62061ac41a40a9002118f24199a677df66"

CF_FIELDS = (
    "production_case_id", "source_views", "source_case_instance_sha256",
    "source_artifacts", "node_pair", "original_topology", "variants",
    "edited_edge", "edit_direction", "original_edge_weight",
    "stay_edge_weight", "switch_edge_weight", "designated_old_path",
    "designated_new_path", "reference_paths", "solver_costs",
    "path_costs", "optimality_relations",
)
RANK_FIELDS = (
    "production_case_id", "eligibility_category", "counterfactual_available",
    "original_graph", "switch_graph", "edited_edge", "old_edge_weight",
    "new_edge_weight", "old_path", "new_path", "path_costs",
    "solver_costs", "preference_relations", "source_artifacts",
)
BOUNDARY_FIELDS = (
    "production_case_id", "eligibility_category", "original_graph",
    "local_stay_graph", "local_switch_graph", "edited_edge",
    "original_edge_weight", "stay_edge_weight", "switch_edge_weight",
    "boundary_tie_weight", "original_weight_vocabulary",
    "old_path", "new_path", "path_costs", "solver_costs",
    "optimality_relations", "boundary_constraints", "source_artifacts",
)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _graph(topology):
    if not isinstance(topology, dict) or not topology:
        raise DatasetFormatError("Expected non-empty topology object")
    graph = {}
    for raw_u, neighbours in topology.items():
        try:
            u = int(raw_u)
            graph[u] = {int(v): int(w) for v, w in neighbours}
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError("Malformed topology") from exc
    for u, neighbours in graph.items():
        for v, weight in neighbours.items():
            if u == v or weight <= 0 or graph.get(v, {}).get(u) != weight:
                raise DatasetFormatError("Invalid weighted undirected graph")
    return graph


def _task_from_prompt(prompt):
    if not isinstance(prompt, str):
        raise DatasetFormatError("Expected historical task prompt")
    match = re.search(
        r"(?m)^Topology: (\{[^\n]+\})\s*$.*?^Node pair: (\[[^\n]+\])",
        prompt,
        re.S,
    )
    if not match:
        raise DatasetFormatError("Unrecognized frozen task prompt format")
    try:
        return _graph(ast.literal_eval(match.group(1))), list(
            map(int, ast.literal_eval(match.group(2)))
        )
    except (ValueError, SyntaxError, TypeError) as exc:
        raise DatasetFormatError("Malformed frozen task prompt") from exc


def _path_cost(graph, path, pair):
    if (
        not isinstance(path, list) or not path
        or len(path) != len(set(path))
        or path[0] != pair[0] or path[-1] != pair[1]
    ):
        raise DatasetFormatError("Invalid simple source-target path")
    try:
        return sum(graph[u][v] for u, v in zip(path, path[1:]))
    except KeyError as exc:
        raise DatasetFormatError("Path uses an absent graph edge") from exc


def _shortest(graph, pair):
    source, target = pair
    queue = [(0, source)]
    distances = {source: 0}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == target:
            return cost
        for neighbour, weight in graph[node].items():
            candidate = cost + weight
            if candidate < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    raise DatasetFormatError("Source and target are disconnected")


def _jsonl(path, *, unique_key="production_case_id"):
    return list(iter_jsonl(path, unique_key=unique_key))


def _write(path, records):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return _sha(path)


def _check_fingerprint(path, expected, name):
    if _sha(path) != expected:
        raise DatasetFormatError(f"{name} frozen SHA-256 mismatch")


def _check_allowlist(record, fields, name):
    if set(record) != set(fields):
        raise DatasetFormatError(f"{name} output fields differ from allowlist")


def _core_g09(path):
    records = _jsonl(path)
    result = {}
    for row in records:
        result[row["production_case_id"]] = row
    return result


def adapt_counterfactual_triples(*, input_path, g09_balanced_path,
                                 g09_train_dist_path, expected_records=1062,
                                 expected_cases=354,
                                 expected_sha256=CF_SHA256,
                                 g09_balanced_sha256=G09_BALANCED_EXPORT_SHA256,
                                 g09_train_dist_sha256=G09_TRAINDIST_EXPORT_SHA256):
    """Validate and normalize the frozen Original/Stay/Switch inputs."""
    _check_fingerprint(input_path, expected_sha256, "Counterfactual input")
    _check_fingerprint(g09_balanced_path, g09_balanced_sha256, "G09 Balanced parent")
    _check_fingerprint(g09_train_dist_path, g09_train_dist_sha256, "G09 TrainDist parent")
    # This source has three rows per parent ID; uniqueness is checked on
    # the parent/variant pair below.
    raw = _jsonl(input_path, unique_key=None)
    if len(raw) != expected_records:
        raise DatasetFormatError("Unexpected counterfactual request count")
    parents = {
        "G09_Balanced": _core_g09(g09_balanced_path),
        "G09_TrainDist": _core_g09(g09_train_dist_path),
    }
    grouped = defaultdict(dict)
    for row in raw:
        cid = row.get("production_case_id")
        if not isinstance(cid, str) or row.get("case_id") != cid:
            raise DatasetFormatError("Counterfactual parent ID mismatch")
        if row.get("counterfactual_parent_case_id") != cid or row.get("condition") != "task_only":
            raise DatasetFormatError("Unexpected counterfactual parent/condition")
        variant = row.get("counterfactual_variant")
        if variant not in ("original", "stay", "switch") or variant in grouped[cid]:
            raise DatasetFormatError("Duplicate or invalid counterfactual variant")
        grouped[cid][variant] = row
    if len(grouped) != expected_cases:
        raise DatasetFormatError("Unexpected unique counterfactual case count")
    result = []
    for cid in sorted(grouped):
        variants = grouped[cid]
        if set(variants) != {"original", "stay", "switch"}:
            raise DatasetFormatError(f"Incomplete counterfactual triple: {cid}")
        source_views = sorted(variants["original"]["source_views"])
        if any(sorted(row.get("source_views", [])) != source_views for row in variants.values()):
            raise DatasetFormatError(f"Counterfactual view membership differs: {cid}")
        if any(row.get("source_case_instance_sha256") != variants["original"].get("source_case_instance_sha256") for row in variants.values()):
            raise DatasetFormatError(f"Counterfactual source identity differs: {cid}")
        if not source_views or any(v not in parents for v in source_views):
            raise DatasetFormatError("Invalid source-view membership")
        parent_rows = [parents[v].get(cid) for v in source_views]
        if any(row is None for row in parent_rows):
            raise DatasetFormatError(f"Missing G09 parent: {cid}")
        original = variants["original"]
        pair = list(map(int, original["node_pair"]))
        graphs = {}
        for name, row in variants.items():
            prompt_graph, prompt_pair = _task_from_prompt(row["messages"][0]["content"])
            payload_graph = _graph(row["topology"])
            if prompt_graph != payload_graph or prompt_pair != pair or row["node_pair"] != pair:
                raise DatasetFormatError(f"Prompt/task disagreement: {cid}")
            graphs[name] = payload_graph
        original_nodes = set(graphs["original"])
        original_edges = {u: set(a) for u, a in graphs["original"].items()}
        if any(
            set(graphs[v]) != original_nodes
            or any(set(graphs[v][u]) != original_edges[u] for u in original_nodes)
            for v in ("stay", "switch")
        ):
            raise DatasetFormatError(f"Counterfactual topology changed: {cid}")
        parent = parent_rows[0]
        parent_graph = _graph(parent["model_input"]["topology"])
        if graphs["original"] != parent_graph:
            raise DatasetFormatError(f"Original graph differs from G09 parent: {cid}")
        for other in parent_rows[1:]:
            if other["model_input"]["topology"] != parent["model_input"]["topology"]:
                raise DatasetFormatError(f"G09 parent views disagree: {cid}")
        edge = tuple(sorted(map(int, original["edited_edge"])))
        if edge[0] not in graphs["original"] or edge[1] not in graphs["original"][edge[0]]:
            raise DatasetFormatError(f"Missing edited edge: {cid}")
        old_weight = graphs["original"][edge[0]][edge[1]]
        edits = {}
        for variant in ("stay", "switch"):
            if tuple(sorted(map(int, variants[variant]["edited_edge"]))) != edge:
                raise DatasetFormatError(f"Edited edge differs across variants: {cid}")
            changed = []
            for u, adj in graphs["original"].items():
                for v, w in adj.items():
                    if u < v and graphs[variant][u][v] != w:
                        changed.append(((u, v), w, graphs[variant][u][v]))
            if len(changed) != 1 or changed[0][0] != edge:
                raise DatasetFormatError(f"Unexpected {variant} edge edits: {cid}")
            edits[variant] = changed[0][2]
        if edits["stay"] == edits["switch"]:
            raise DatasetFormatError(f"Stay/Switch weights are not distinct: {cid}")
        direction = original["edit_direction"]
        if direction == "increase":
            direction_valid = edits["stay"] > old_weight and edits["switch"] > old_weight
        elif direction == "decrease":
            direction_valid = edits["stay"] < old_weight and edits["switch"] < old_weight
        else:
            direction_valid = False
        if not direction_valid:
            raise DatasetFormatError(f"Recorded edge edit direction mismatch: {cid}")
        old_path = list(map(int, original["gold_path"]))
        new_path = list(map(int, variants["switch"]["gold_path"]))
        costs = {v: int(variants[v]["gold_cost"]) for v in variants}
        path_costs = {
            "original_old": _path_cost(graphs["original"], old_path, pair),
            "original_new": _path_cost(graphs["original"], new_path, pair),
            "stay_old": _path_cost(graphs["stay"], old_path, pair),
            "switch_old": _path_cost(graphs["switch"], old_path, pair),
            "switch_new": _path_cost(graphs["switch"], new_path, pair),
        }
        solvers = {v: _shortest(graphs[v], pair) for v in graphs}
        if any(solvers[v] != costs[v] for v in variants):
            raise DatasetFormatError(f"Solver reference mismatch: {cid}")
        for variant in ("original", "stay", "switch"):
            reference_path = list(map(int, variants[variant]["gold_path"]))
            if _path_cost(graphs[variant], reference_path, pair) != costs[variant]:
                raise DatasetFormatError(f"Reference path is not optimal: {cid}")
        if path_costs["original_old"] != costs["original"] or path_costs["stay_old"] != costs["stay"]:
            raise DatasetFormatError(f"Original designated path not retained: {cid}")
        if path_costs["switch_old"] <= solvers["switch"] or path_costs["switch_new"] != solvers["switch"]:
            raise DatasetFormatError(f"Switch optimality relation mismatch: {cid}")
        for parent_row in parent_rows:
            if (
                parent_row["provenance"]["case_instance_sha256"] != original["source_case_instance_sha256"]
                or parent_row["reference"]["ground_truth_shortest_path"] != original["ground_truth_shortest_path"]
                or parent_row["reference"]["gold_cost"] != original["gold_cost"]
            ):
                raise DatasetFormatError(f"G09 parent reference mismatch: {cid}")
        record = {
            "production_case_id": cid,
            "source_views": source_views,
            "source_case_instance_sha256": original["source_case_instance_sha256"],
            "source_artifacts": {"evaluation_input_sha256": expected_sha256,
                                 "construction_script_sha256": CF_SCRIPT_SHA256,
                                 "g09_balanced_sha256": G09_BALANCED_SHA256,
                                 "g09_traindist_sha256": G09_TRAINDIST_SHA256},
            "node_pair": pair,
            "original_topology": original["topology"],
            "variants": {v: {"topology": variants[v]["topology"], "gold_cost": costs[v]} for v in ("original", "stay", "switch")},
            "edited_edge": list(edge),
            "edit_direction": original["edit_direction"],
            "original_edge_weight": old_weight,
            "stay_edge_weight": edits["stay"],
            "switch_edge_weight": edits["switch"],
            "designated_old_path": old_path,
            "designated_new_path": new_path,
            "reference_paths": {v: variants[v]["gold_path"] for v in variants},
            "solver_costs": solvers,
            "path_costs": path_costs,
            "optimality_relations": {"original_old_optimal": True, "stay_old_optimal": True, "switch_old_strictly_suboptimal": True, "switch_new_optimal": True},
        }
        _check_allowlist(record, CF_FIELDS, "Counterfactual")
        result.append(record)
    return result


def _eligibility_record(row, entry, core_g06_sha256=None):
    cid = row["production_case_id"]
    available = row["cf_available"] is True
    if available != (entry["mode"] != "SFT_ONLY"):
        raise DatasetFormatError(f"Counterfactual availability mismatch: {cid}")
    if not available:
        original, pair = _task_from_prompt(row["prompt"])
        old = list(map(int, row["old_path"]))
        original_old = _path_cost(original, old, pair)
        original_solver = _shortest(original, pair)
        if original_old != row["old_optimal_cost"] or original_solver != original_old:
            raise DatasetFormatError(f"Unavailable ranking source path/solver mismatch: {cid}")
        record = {
            "production_case_id": cid, "eligibility_category": "no_strict_switch",
            "counterfactual_available": False,
            "original_graph": {"topology": {str(u): [[v,w] for v,w in sorted(a.items())] for u,a in original.items()}, "node_pair": pair},
            "switch_graph": None, "edited_edge": None, "old_edge_weight": None,
            "new_edge_weight": None, "old_path": row["old_path"], "new_path": None,
            "path_costs": {"original_old": original_old},
            "solver_costs": {"original": original_solver, "switch": None},
            "preference_relations": {"original": "not_ranked", "switch": "not_available"},
            "source_artifacts": {"ranking_sha256": RANK_SHA256, "eligibility_sha256": ELIGIBILITY_SHA256, "core_g06_train_sha256": core_g06_sha256, "construction_script_sha256": RANK_SCRIPT_SHA256},
        }
        _check_allowlist(record, RANK_FIELDS, "Ranking")
        return record
    original, pair = _task_from_prompt(row["prompt"])
    switch, switch_pair = _task_from_prompt(row["cf_prompt"])
    if pair != switch_pair:
        raise DatasetFormatError(f"Ranking node pair mismatch: {cid}")
    old = list(map(int, row["old_path"]))
    new = list(map(int, row["cf_optimal_path"]))
    cgo = _path_cost(original, old, pair); cgn = _path_cost(original, new, pair)
    cpo = _path_cost(switch, old, pair); cpn = _path_cost(switch, new, pair)
    so = _shortest(original, pair); sp = _shortest(switch, pair)
    if (cgo != row["old_optimal_cost"] or cpo != row["old_path_cost_on_cf"] or cpn != row["cf_optimal_cost"] or so != row["old_optimal_cost"] or sp != row["cf_optimal_cost"]):
        raise DatasetFormatError(f"Ranking path/solver cost mismatch: {cid}")
    mode = "strict_bidirectional" if cgo < cgn and cpn < cpo else "original_tie_switch_strict" if cgo == cgn and cpn < cpo else "no_strict_switch"
    if mode != {"BIDIRECTIONAL":"strict_bidirectional", "SWITCH_ONLY_ORIGINAL_TIE":"original_tie_switch_strict", "SFT_ONLY":"no_strict_switch"}[entry["mode"]]:
        raise DatasetFormatError(f"Ranking preference classification mismatch: {cid}")
    edge = list(map(int, row["edited_edge"])); u,v=sorted(edge)
    oldw=int(row["old_edge_weight"]); neww=int(row["new_edge_weight"])
    expected={x:dict(a) for x,a in original.items()};expected[u][v]=expected[v][u]=neww
    if expected!=switch or original[u][v]!=oldw:
        raise DatasetFormatError(f"Ranking graph edit mismatch: {cid}")
    record = {
        "production_case_id": cid, "eligibility_category": mode,
        "counterfactual_available": True,
        "original_graph": {"topology": {str(u): [[v,w] for v,w in sorted(a.items())] for u,a in original.items()}, "node_pair": pair},
        "switch_graph": {"topology": {str(u): [[v,w] for v,w in sorted(a.items())] for u,a in switch.items()}, "node_pair": pair}, "edited_edge": edge,
        "old_edge_weight": oldw, "new_edge_weight": neww,
        "old_path": old, "new_path": new,
        "path_costs": {"original_old":cgo,"original_new":cgn,"switch_old":cpo,"switch_new":cpn},
        "solver_costs": {"original":so,"switch":sp},
        "preference_relations": {"original":"old_strictly_preferred" if cgo<cgn else "tie", "switch":"new_strictly_preferred"},
        "source_artifacts": {"ranking_sha256":RANK_SHA256,"eligibility_sha256":ELIGIBILITY_SHA256,"core_g06_train_sha256":core_g06_sha256,"construction_script_sha256":RANK_SCRIPT_SHA256},
    }
    _check_allowlist(record, RANK_FIELDS, "Ranking")
    return record


def adapt_ranking_eligibility(*, ranking_path, eligibility_path, core_g06_train_path,
                              expected_records=200, ranking_sha256=RANK_SHA256,
                              eligibility_sha256=ELIGIBILITY_SHA256,
                              core_g06_sha256=G06_TRAIN_EXPORT_SHA256):
    _check_fingerprint(ranking_path, ranking_sha256, "Ranking source")
    _check_fingerprint(eligibility_path, eligibility_sha256, "Eligibility source")
    _check_fingerprint(core_g06_train_path, core_g06_sha256, "Core G06 Train-200 parent")
    ranks=_jsonl(ranking_path); eligibility=_jsonl(eligibility_path); core=_jsonl(core_g06_train_path)
    if len(ranks)!=expected_records or len(eligibility)!=expected_records or len(core)!=expected_records:
        raise DatasetFormatError("Unexpected G06 ranking/core count")
    core_ids={r["production_case_id"] for r in core}
    by_id={r["production_case_id"]:r for r in eligibility}
    if len(by_id)!=len(eligibility) or {r["production_case_id"] for r in ranks}!=core_ids or set(by_id)!=core_ids:
        raise DatasetFormatError("Ranking IDs do not exactly match G06 Train-200")
    core_sha256 = _sha(core_g06_train_path)
    output=[_eligibility_record(r,by_id[r["production_case_id"]],core_sha256) for r in ranks]
    output.sort(key=lambda record: record["production_case_id"])
    for record in output:
        _check_allowlist(record, RANK_FIELDS, "Ranking")
    counts={k:sum(r["eligibility_category"]==k for r in output) for k in ("strict_bidirectional","original_tie_switch_strict","no_strict_switch")}
    if counts!={"strict_bidirectional":162,"original_tie_switch_strict":9,"no_strict_switch":29}:
        raise DatasetFormatError(f"Unexpected ranking eligibility partition: {counts}")
    return output,counts


def adapt_boundary_local(*, boundary_path, eligibility_records, protocol_path,
                         expected_records=162, boundary_sha256=BOUNDARY_SHA256,
                         protocol_sha256=BOUNDARY_PROTOCOL_SHA256):
    _check_fingerprint(boundary_path,boundary_sha256,"Boundary-Local source")
    _check_fingerprint(protocol_path,protocol_sha256,"Boundary-Local protocol")
    protocol=json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if (protocol.get("coverage",{}).get("strict_boundary_local")!=expected_records
        or protocol.get("objective",{}).get("margin")!=1.0
        or protocol.get("canonical_source_script_sha256") != BOUNDARY_TRAIN_SCRIPT_SHA256):
        raise DatasetFormatError("Unexpected Boundary-Local protocol")
    strict_records={r["production_case_id"]:r for r in eligibility_records if r["eligibility_category"]=="strict_bidirectional"}
    strict=set(strict_records)
    rows=_jsonl(boundary_path, unique_key=None)
    if len(rows)!=expected_records or len({r["case_id"] for r in rows})!=len(rows) or {r["case_id"] for r in rows}!=strict:
        raise DatasetFormatError("Boundary-Local IDs do not match strict G06 subset")
    result=[]
    for r in rows:
        cid=r["case_id"]; pair=None; graphs={}
        for variant in ("original","stay","switch"):
            graphs[variant],p=_task_from_prompt(r[variant+"_prompt"])
            if pair is None:pair=p
            elif p!=pair:raise DatasetFormatError(f"Boundary node-pair mismatch: {cid}")
        edge=list(map(int,r["edited_edge"]));u,v=sorted(edge)
        old=list(map(int,r["old_path"]));new=list(map(int,r["new_path"]))
        parent = strict_records[cid]
        boundary_original = {"topology": {str(a): [[b, w] for b, w in sorted(adj.items())] for a, adj in graphs["original"].items()}, "node_pair": pair}
        if (
            boundary_original != parent["original_graph"]
            or old != parent["old_path"]
            or new != parent["new_path"]
            or sorted(edge) != sorted(parent["edited_edge"])
            or int(r["original_edge_weight"]) != parent["old_edge_weight"]
        ):
            raise DatasetFormatError(f"Boundary parent graph/path mismatch: {cid}")
        if any(set(g)!=set(graphs['original']) for g in graphs.values()):raise DatasetFormatError("Boundary topology changed")
        for variant,key in (("original","original_edge_weight"),("stay","stay_edge_weight"),("switch","switch_edge_weight")):
            if graphs[variant][u][v]!=r[key]:raise DatasetFormatError(f"Boundary edited weight mismatch: {cid}")
            if any(set(graphs[variant][x])!=set(graphs['original'][x]) for x in graphs['original']):raise DatasetFormatError("Boundary edge set changed")
            if any(
                graphs[variant][x][y] != graphs["original"][x][y]
                for x in graphs["original"] for y in graphs["original"][x]
                if (x, y) not in ((u, v), (v, u))
            ):
                raise DatasetFormatError(f"Boundary unrelated edge weight changed: {cid}")
        costs={f"{v}_{n}":_path_cost(graphs[v],path,pair) for v in graphs for n,path in (("old",old),("new",new))}
        solvers={v:_shortest(g,pair) for v,g in graphs.items()}
        if costs["original_old"]!=r["original_old_cost"] or costs["original_new"]!=r["original_new_cost"] or costs["stay_old"]!=r["stay_old_cost"] or costs["stay_new"]!=r["stay_new_cost"] or costs["switch_old"]!=r["switch_old_cost"] or costs["switch_new"]!=r["switch_new_cost"]:
            raise DatasetFormatError(f"Boundary path cost mismatch: {cid}")
        if solvers["stay"]!=r["stay_solver_optimal_cost"] or solvers["switch"]!=r["switch_solver_optimal_cost"]:
            raise DatasetFormatError(f"Boundary solver cost mismatch: {cid}")
        if (
            _path_cost(graphs["stay"], r["stay_solver_path"], pair) != solvers["stay"]
            or _path_cost(graphs["switch"], r["switch_solver_path"], pair) != solvers["switch"]
        ):
            raise DatasetFormatError(f"Boundary solver reference path mismatch: {cid}")
        original_vocabulary = {w for a in graphs["original"].values() for w in a.values()}
        stay_in_vocab = int(r["stay_edge_weight"]) in original_vocabulary
        switch_in_vocab = int(r["switch_edge_weight"]) in original_vocabulary
        if (
            set(r["original_weight_vocabulary"]) != original_vocabulary
            or r["stay_weight_in_original_vocabulary"] is not stay_in_vocab
            or r["switch_weight_in_original_vocabulary"] is not switch_in_vocab
            or r["both_local_weights_in_original_vocabulary"] is not (stay_in_vocab and switch_in_vocab)
            or r["old_uses_edited_edge"] is not (any(tuple(sorted((a, b))) == (u, v) for a, b in zip(old, old[1:])))
            or r["new_uses_edited_edge"] is not (any(tuple(sorted((a, b))) == (u, v) for a, b in zip(new, new[1:])))
        ):
            raise DatasetFormatError(f"Boundary constraint metadata mismatch: {cid}")
        if (not r["stay_designated_old_is_global_optimum"]
            or not r["switch_designated_new_is_global_optimum"]
            or costs["original_old"] >= costs["original_new"]
            or costs["original_new"] - costs["original_old"] != r["original_pair_gap_new_minus_old"]
            or costs["stay_new"] - costs["stay_old"] != 1
            or costs["switch_old"] - costs["switch_new"] != 1
            or {int(r["stay_edge_weight"]) - int(r["boundary_tie_weight"]), int(r["switch_edge_weight"]) - int(r["boundary_tie_weight"])} != {-1, 1}
            or solvers["original"] != costs["original_old"]
        ):
            raise DatasetFormatError(f"Boundary preference relation mismatch: {cid}")
        record = {
            "production_case_id":cid,"eligibility_category":"strict_bidirectional",
            "original_graph":{"topology":{str(u):[[v,w] for v,w in sorted(a.items())] for u,a in graphs["original"].items()},"node_pair":pair},
            "local_stay_graph":{"topology":{str(u):[[v,w] for v,w in sorted(a.items())] for u,a in graphs["stay"].items()},"node_pair":pair},
            "local_switch_graph":{"topology":{str(u):[[v,w] for v,w in sorted(a.items())] for u,a in graphs["switch"].items()},"node_pair":pair},
            "edited_edge":edge,"original_edge_weight":r["original_edge_weight"],"stay_edge_weight":r["stay_edge_weight"],"switch_edge_weight":r["switch_edge_weight"],
            "boundary_tie_weight":r["boundary_tie_weight"],"original_weight_vocabulary":r["original_weight_vocabulary"],
            "old_path":old,"new_path":new,"path_costs":costs,"solver_costs":solvers,
            "optimality_relations":{"original_old_strictly_better":True,"stay_old_strictly_better":True,"switch_new_strictly_better":True},
            "boundary_constraints":{"both_local_weights_in_original_vocabulary":r["both_local_weights_in_original_vocabulary"],"stay_weight_in_original_vocabulary":r["stay_weight_in_original_vocabulary"],"switch_weight_in_original_vocabulary":r["switch_weight_in_original_vocabulary"],"stay_switch_are_opposite_sides_of_tie":True,"old_path_uses_edited_edge":r["old_uses_edited_edge"],"new_path_uses_edited_edge":r["new_uses_edited_edge"]},
            "source_artifacts":{"boundary_sha256":boundary_sha256,"protocol_sha256":_sha(protocol_path),"construction_script_sha256":BOUNDARY_BUILDER_SHA256,"canonical_training_script_sha256":protocol["canonical_source_script_sha256"]},
        }
        _check_allowlist(record, BOUNDARY_FIELDS, "Boundary-Local")
        result.append(record)
    result.sort(key=lambda record: record["production_case_id"])
    return result


def export_derived_candidate(*, counterfactual_path, g09_balanced_path,
                             g09_train_dist_path, ranking_path,
                             eligibility_path, core_g06_train_path,
                             boundary_path, boundary_protocol_path,
                             counterfactual_script_path,
                             ranking_script_path, boundary_script_path,
                             scoring_report_path,
                             output_dir, dataset_version="pathfact-derived-v0.1",
                             expected_counterfactual_sha256=CF_SHA256,
                             expected_ranking_sha256=RANK_SHA256,
                             expected_eligibility_sha256=ELIGIBILITY_SHA256,
                             expected_boundary_sha256=BOUNDARY_SHA256):
    """Build a private draft candidate from the completed archived resources."""
    output=Path(output_dir)
    if output.exists() or output.is_symlink():raise FileExistsError(f"Output already exists: {output}")
    if not output.parent.is_dir():raise FileNotFoundError("Output parent must exist")
    sources={"counterfactual":Path(counterfactual_path),"g09_balanced":Path(g09_balanced_path),"g09_train_dist":Path(g09_train_dist_path),"ranking":Path(ranking_path),"eligibility":Path(eligibility_path),"core_g06_train":Path(core_g06_train_path),"boundary":Path(boundary_path),"boundary_protocol":Path(boundary_protocol_path),"counterfactual_script":Path(counterfactual_script_path),"ranking_script":Path(ranking_script_path),"boundary_script":Path(boundary_script_path),"scoring_report":Path(scoring_report_path)}
    before={k:_sha(v) for k,v in sources.items()}
    for key, expected, name in (
        ("counterfactual_script", CF_SCRIPT_SHA256, "Counterfactual construction script"),
        ("ranking_script", RANK_SCRIPT_SHA256, "Ranking construction script"),
        ("boundary_script", BOUNDARY_BUILDER_SHA256, "Boundary-Local construction script"),
    ):
        if before[key] != expected:
            raise DatasetFormatError(f"{name} SHA-256 mismatch")
    cf=adapt_counterfactual_triples(input_path=sources["counterfactual"],g09_balanced_path=sources["g09_balanced"],g09_train_dist_path=sources["g09_train_dist"],expected_sha256=expected_counterfactual_sha256)
    ranking,partition=adapt_ranking_eligibility(ranking_path=sources["ranking"],eligibility_path=sources["eligibility"],core_g06_train_path=sources["core_g06_train"],ranking_sha256=expected_ranking_sha256,eligibility_sha256=expected_eligibility_sha256)
    boundary=adapt_boundary_local(boundary_path=sources["boundary"],eligibility_records=ranking,protocol_path=sources["boundary_protocol"],boundary_sha256=expected_boundary_sha256)
    scoring=json.loads(sources["scoring_report"].read_text(encoding="utf-8"))
    if scoring.get("input_sha256")!=expected_counterfactual_sha256:
        raise DatasetFormatError("October scoring report input fingerprint mismatch")
    model_summaries=scoring.get("model_summaries",{})
    if len(model_summaries)!=8 or any(summary.get("n")!=1062 for summary in model_summaries.values()):
        raise DatasetFormatError("Unexpected October scoring condition/response counts")
    previous_umask = os.umask(0o077)
    staging = None
    try:
        staging=Path(tempfile.mkdtemp(prefix=".pathfact-derived-",dir=output.parent))
        for d in (staging/"counterfactual",staging/"training",staging/"metadata"):d.mkdir()
        artifacts=[]
        for name,records,rel,stage,schema in (("counterfactual",cf,"counterfactual/counterfactual_triples.jsonl","G09-CF",CF_SCHEMA),("ranking",ranking,"training/ranking_eligibility.jsonl","G06-Ranking",RANK_SCHEMA),("boundary",boundary,"training/boundary_local.jsonl","G06-BoundaryLocal",BOUNDARY_SCHEMA)):
            digest=_write(staging/rel,records)
            source_fingerprints = {
                "counterfactual": {
                    "evaluation_input": before["counterfactual"],
                    "construction_script": before["counterfactual_script"],
                    "g09_balanced": before["g09_balanced"],
                    "g09_train_dist": before["g09_train_dist"],
                },
                "ranking": {
                    "ranking": before["ranking"],
                    "construction_script": before["ranking_script"],
                    "eligibility": before["eligibility"],
                    "core_g06_train": before["core_g06_train"],
                },
                "boundary": {
                    "boundary": before["boundary"],
                    "construction_script": before["boundary_script"],
                    "protocol": before["boundary_protocol"],
                },
            }
            artifacts.append({"path":rel,"source_stage":stage,"schema_version":schema,"source_sha256":source_fingerprints[name]})
        manifest_path=create_release_manifest(staging,artifacts,dataset_version)
        verification=verify_release_manifest(staging)
        summary={
            "schema_version": "pathfact-derived-candidate-v0.1",
            "dataset_version": dataset_version,
            "release_status": "draft_not_approved",
            "license": None,
            "artifact_schemas": {
                "counterfactual_triples": CF_SCHEMA,
                "ranking_eligibility": RANK_SCHEMA,
                "boundary_local": BOUNDARY_SCHEMA,
            },
            "counts": {
                "counterfactual_triples": len(cf),
                "counterfactual_view_memberships": sum(len(r["source_views"]) for r in cf),
                "ranking_eligibility": len(ranking),
                "ranking_partition": partition,
                "boundary_local": len(boundary),
            },
            "parent_relationships": {
                "counterfactual_parent": "G09 canonical evaluation views",
                "ranking_parent": "G06 Gold Train-200",
                "boundary_local_parent": "strict_bidirectional ranking subset",
            },
            "source_sha256": before,
            "counterfactual_scoring": {
                "report_sha256": before["scoring_report"],
                "input_sha256": scoring["input_sha256"],
                "condition_count": len(model_summaries),
                "requests_per_condition": 1062,
                "recorded_response_count": sum(x["n"] for x in model_summaries.values()),
            },
            "validation_results": {
                "counterfactual_parent_and_graph_checks": "PASS",
                "ranking_identity_and_eligibility_checks": "PASS",
                "boundary_parent_and_protocol_checks": "PASS",
                "manifest_verification": verification["status"],
                "source_unchanged_after_build": "PASS",
            },
            "manifest_sha256": _sha(manifest_path),
            "manifest_verification": verification,
        }
        (staging/"metadata/summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        if any(_sha(p)!=before[k] for k,p in sources.items()):raise DatasetFormatError("A source changed during derived export")
        for p in staging.rglob("*"):p.chmod(0o700 if p.is_dir() else 0o600)
        staging.chmod(0o700)
        if output.exists() or output.is_symlink():raise FileExistsError(f"Output appeared during export: {output}")
        staging.rename(output)
    except Exception:
        if staging is not None:
            shutil.rmtree(staging,ignore_errors=True)
        raise
    finally:
        os.umask(previous_umask)
    return {"root":output,"manifest":output/"manifest.json","summary":output/"metadata/summary.json","counterfactual":output/"counterfactual/counterfactual_triples.jsonl","ranking":output/"training/ranking_eligibility.jsonl","boundary":output/"training/boundary_local.jsonl"}
