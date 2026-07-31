"""
Epistemic Status — Blix v0.3.11  (cross-cutting infrastructure)

Every object introduced in v0.3.11 — and, optionally, existing objects
like ``memory.beliefs.Belief`` — can now carry an explicit
``EpistemicStatus``: not "is this currently true" (that's
``core.truth_manager.TruthStatus``, a v0.3.7 concern, orthogonal to
this), but "HOW did Blix come to hold this, and how much should it be
trusted as a result."

    OBSERVED        — directly witnessed (a completed task, a stated user fact)
    DERIVED         — computed/inferred from OBSERVED data (e.g. a CauseGraph edge from co-occurrence counts)
    PREDICTED       — a forward-looking estimate (e.g. world_model.latent_world_model output)
    COUNTERFACTUAL  — "what if" — an alternate-scenario estimate, explicitly NOT a claim about reality
    PRINCIPLE       — a synthesized, reusable generalization derived from multiple DERIVED/OBSERVED facts
    HYPOTHESIS      — a candidate belief awaiting confirming observation; not yet trusted as OBSERVED

This module enforces nothing by itself — it's a vocabulary. The actual
safeguard (COUNTERFACTUAL must never become OBSERVED implicitly) is
enforced structurally: ``causality.counterfactual_engine`` does not
import ``memory.beliefs`` at all, and ``memory.beliefs.BeliefStore``'s
only path from a low-trust status to a trusted one is the explicit,
separate ``confirm_observation()`` call — there is no single function
that can take a COUNTERFACTUAL result and silently turn it into an
OBSERVED belief.

Python 3.10 compatible.
"""
# DEPRECATED — causality.epistemic_status (ISSUE-009)
#
# This module is superseded by memory.hybrid.models.memory_node.
# The class ``EpistemicStatus`` here is the v0.3.x implementation;
# ``memory.hybrid.models.memory_node.EpistemicStatus`` is the v0.3.15+ HGSHM implementation.
#
# These are different classes with different APIs. Callers that need
# the v0.3.15+ version must update their imports:
#
#     # Old (this file — legacy):
#     from causality.epistemic_status import EpistemicStatus
#
#     # New (HGSHM-backed):
#     from memory.hybrid.models.memory_node import EpistemicStatus
#
# This file will be removed in v0.4. Do not add new callers.
# Issue: https://github.com/blix/blix/issues/9
#


from __future__ import annotations

from enum import Enum


class EpistemicStatus(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    PREDICTED = "predicted"
    COUNTERFACTUAL = "counterfactual"
    PRINCIPLE = "principle"
    HYPOTHESIS = "hypothesis"

    @property
    def is_trusted(self) -> bool:
        """
        Whether this status alone is sufficient grounds to treat the
        object as a reliable fact. OBSERVED and DERIVED (computed
        directly from observed data) are trusted; everything else
        (predictions, counterfactuals, unconfirmed hypotheses, and
        principles awaiting validation) is not.
        """
        return self in (EpistemicStatus.OBSERVED, EpistemicStatus.DERIVED)

    @property
    def requires_validation_before_belief(self) -> bool:
        """
        Whether promoting this object toward a trusted Belief requires
        an explicit confirming-observation step rather than being
        usable as-is. True for everything except OBSERVED/DERIVED.
        """
        return not self.is_trusted
