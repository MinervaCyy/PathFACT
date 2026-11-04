"""Weighted shortest-path solving and hop-aware task selection."""

from .canonical import (
    eligible_pairs_by_exact_hop,
    reconstruct_path,
    select_pair_exact_hop_aware,
    shortest_path_hop_bounds_from_source,
)
