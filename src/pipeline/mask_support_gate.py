"""Optional visible-mask evidence gate for heuristic support proposals."""

from __future__ import annotations

import numpy as np

from .mask_contact import visible_contact_fraction


def gate_support_proposals(proposals: list[dict], masks: dict[str, np.ndarray],
                           min_contact_fraction: float = 0.0) -> tuple[list[dict], list[dict]]:
    """Retain only proposals with visible mask adjacency above the threshold.

    This is a conservative image-space heuristic. A rejected pair may still be
    physically supported when the contact area is occluded or masks are poor.
    """
    if not np.isfinite(min_contact_fraction) or not 0 <= min_contact_fraction < 1:
        raise ValueError("min_contact_fraction must be in [0, 1)")
    accepted, evidence = [], []
    for proposal in proposals:
        child_id = str(proposal["child_id"])
        parent_id = str(proposal["parent_id"])
        if child_id not in masks or parent_id not in masks:
            raise ValueError("Each proposed object needs a mask")
        contact = visible_contact_fraction(masks[child_id], masks[parent_id])
        passed = contact["sampled_columns"] > 0 and contact["contact_fraction"] > min_contact_fraction
        evidence.append({"child_id": child_id, "parent_id": parent_id,
                         **contact, "accepted": passed})
        if passed:
            accepted.append({**proposal, "visible_mask_contact_fraction": contact["contact_fraction"]})
    return accepted, evidence
