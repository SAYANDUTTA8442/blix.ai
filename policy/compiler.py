"""
PolicySelector — context-aware policy selection with fallback logic.
PolicyCompiler — assembles dynamic prompts from system + user policies + context.

The Compiler replaces the static system prompt with a dynamically
assembled instruction set derived from:
  1. System policy config (how Blix operates)
  2. User policy config (how Blix behaves for this user)
  3. Retrieved MemoryContext (relevant knowledge)
  4. Task context (what is being asked)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any

from policy.models import PolicyRecord, PolicyDomain, PolicyType
from policy.learner import PolicyLearner

log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# PolicySelector
# ────────────────────────────────────────────────────────────────────

class PolicySelector:
    """
    Context-aware policy selector with graceful fallback.

    Wraps PolicyLearner.select() and adds:
      • Fallback to global defaults when no user-specific policy exists
      • Context enrichment (query length bucketing, topic detection)
      • Multi-type selection (retrieve all needed policies in one call)
    """

    def __init__(self, learner: PolicyLearner) -> None:
        self._learner = learner

    def select_system_policies(
        self, context: dict[str, Any] | None = None
    ) -> dict[PolicyType, PolicyRecord | None]:
        """
        Select the best system policy for each type.

        Returns a mapping of PolicyType → best PolicyRecord.
        """
        context = context or {}
        system_types = [
            PolicyType.RETRIEVAL_WEIGHTS,
            PolicyType.PLANNER_CONFIG,
            PolicyType.REASONING_STRATEGY,
            PolicyType.VERIFICATION_POLICY,
            PolicyType.WORKSPACE_CONFIG,
            PolicyType.COMPRESSION_POLICY,
        ]
        result: dict[PolicyType, PolicyRecord | None] = {}
        for pt in system_types:
            selected = self._learner.select_one(
                pt, domain=PolicyDomain.SYSTEM, context=context)
            result[pt] = selected
        return result

    def select_user_policies(
        self,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> dict[PolicyType, PolicyRecord | None]:
        """
        Select the best user-personalisation policy for each type.

        Falls back to global defaults if no user-specific policy exists.
        """
        context = context or {}
        user_types = [
            PolicyType.ANSWER_STYLE,
            PolicyType.DIFFICULTY_LEVEL,
            PolicyType.EXPLANATION_DEPTH,
            PolicyType.TOPIC_PREFERENCE,
            PolicyType.HINT_POLICY,
            PolicyType.GOAL_PRIORITY,
        ]
        result: dict[PolicyType, PolicyRecord | None] = {}
        for pt in user_types:
            selected = self._learner.select_one(
                pt, domain=PolicyDomain.USER,
                context={**context, "user_id": user_id},
                user_id=user_id,
            )
            result[pt] = selected
        return result

    def get_retrieval_weights(
        self, context: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """
        Return the current best retrieval weight configuration.
        Falls back to balanced weights if no policy exists.
        """
        policy = self._learner.select_one(
            PolicyType.RETRIEVAL_WEIGHTS,
            domain=PolicyDomain.SYSTEM,
            context=context,
        )
        if policy and policy.config:
            return policy.config
        # Fallback: uniform weights
        keys = ["semantic", "vector", "graph_distance", "importance", "confidence",
                "recency", "hierarchy", "context_similarity", "attention",
                "belief_confidence", "planning_relevance"]
        return {k: 1.0 / len(keys) for k in keys}

    def get_planner_config(
        self, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return the current best planner configuration."""
        policy = self._learner.select_one(
            PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM,
            context=context,
        )
        if policy and policy.config:
            return policy.config
        return {"beam_width": 3, "max_depth": 2, "branching": 3}

    def get_answer_style(self, user_id: str,
                          context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the current best answer style for a user."""
        policy = self._learner.select_one(
            PolicyType.ANSWER_STYLE,
            domain=PolicyDomain.USER,
            context=context,
            user_id=user_id,
        )
        if policy and policy.config:
            return policy.config
        return {"verbosity": "med", "code_first": True, "examples": True}


# ────────────────────────────────────────────────────────────────────
# CompiledPrompt
# ────────────────────────────────────────────────────────────────────

@dataclass
class CompiledPrompt:
    """
    The output of PolicyCompiler.compile().

    Instead of a flat string, this is a structured object that callers
    can use to assemble the final prompt in the format their LLM expects.
    """
    system_instructions:  str             = ""
    user_context:         str             = ""
    memory_context:       str             = ""
    task_instructions:    str             = ""
    constraints:          list[str]       = field(default_factory=list)
    active_policies:      dict[str, str]  = field(default_factory=dict)   # type → name
    token_estimate:       int             = 0
    metadata:             dict[str, Any]  = field(default_factory=dict)

    def to_flat_string(self, separator: str = "\n\n") -> str:
        """Assemble a flat prompt string for LLMs that expect one."""
        parts = []
        if self.system_instructions:
            parts.append(self.system_instructions)
        if self.user_context:
            parts.append(self.user_context)
        if self.memory_context:
            parts.append(self.memory_context)
        if self.task_instructions:
            parts.append(self.task_instructions)
        if self.constraints:
            parts.append("Constraints:\n" + "\n".join(f"• {c}" for c in self.constraints))
        return separator.join(parts)

    def to_messages(self) -> list[dict[str, str]]:
        """Assemble as a messages list (OpenAI / Anthropic format)."""
        messages = []
        system_text = self.to_flat_string()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        return messages

    @property
    def total_chars(self) -> int:
        return len(self.to_flat_string())


# ────────────────────────────────────────────────────────────────────
# PolicyCompiler
# ────────────────────────────────────────────────────────────────────

class PolicyCompiler:
    """
    Replaces the static system prompt with a dynamically compiled
    instruction set derived from active policies.

    Pipeline
    --------
    Task + User ID + MemoryContext
      → PolicySelector (select best system + user policies)
      → Instruction Templates (fill policy config into text)
      → Memory Context Formatter (top memories as text)
      → Constraint Assembler (hard constraints from config)
      → CompiledPrompt
    """

    def __init__(self, policy_selector: PolicySelector) -> None:
        self._selector = policy_selector

    def compile(
        self,
        task: str,
        user_id: str = "default",
        memory_context: Any = None,    # memory.hybrid.models.memory_context.MemoryContext
        context: dict[str, Any] | None = None,
        max_memory_nodes: int = 5,
        token_budget: int = 2000,
    ) -> CompiledPrompt:
        """
        Compile a dynamic prompt for the given task and user.

        Parameters
        ----------
        task : str
            The user's request or task description.
        user_id : str
            Identifier for the current user.
        memory_context : MemoryContext | None
            Retrieved memories from HGSHM.
        context : dict
            Additional context features for policy selection.
        max_memory_nodes : int
            Maximum memories to include in the compiled prompt.
        token_budget : int
            Target token budget (used to truncate memory context).
        """
        ctx = {**(context or {}), "task": task[:100], "user_id": user_id}

        # ── 1. Select policies ───────────────────────────────────────
        sys_policies  = self._selector.select_system_policies(ctx)
        user_policies = self._selector.select_user_policies(user_id, ctx)

        active_policies: dict[str, str] = {}
        for pt, p in {**sys_policies, **user_policies}.items():
            if p:
                active_policies[pt.value] = p.name

        # ── 2. Build system instructions from system policies ────────
        sys_parts = ["You are Blix, a cognitive AI assistant."]
        reasoning = sys_policies.get(PolicyType.REASONING_STRATEGY)
        if reasoning and reasoning.config:
            cfg = reasoning.config
            chain_len = cfg.get("chain_length", 3)
            decompose  = cfg.get("decompose", True)
            verify     = cfg.get("verify", True)
            if chain_len > 1:
                sys_parts.append(
                    f"Use {chain_len}-step reasoning chains for complex tasks.")
            if decompose:
                sys_parts.append("Decompose complex problems before solving.")
            if verify:
                sys_parts.append("Verify your answers before presenting them.")

        planner = sys_policies.get(PolicyType.PLANNER_CONFIG)
        if planner and planner.config:
            cfg = planner.config
            bw = cfg.get("beam_width", 3)
            depth = cfg.get("max_depth", 2)
            sys_parts.append(
                f"Planning: consider up to {bw} candidate paths, "
                f"evaluating {depth} steps ahead.")

        system_instructions = "\n".join(sys_parts)

        # ── 3. Build user context from user policies ─────────────────
        user_parts = []
        style = user_policies.get(PolicyType.ANSWER_STYLE)
        if style and style.config:
            cfg = style.config
            verbosity = cfg.get("verbosity", "med")
            code_first = cfg.get("code_first", False)
            examples   = cfg.get("examples", True)
            verb_map = {"low": "concise", "med": "balanced", "high": "detailed"}
            user_parts.append(f"Response style: {verb_map.get(verbosity, 'balanced')}.")
            if code_first:
                user_parts.append("Lead with code, then explain.")
            if examples:
                user_parts.append("Include illustrative examples.")

        difficulty = user_policies.get(PolicyType.DIFFICULTY_LEVEL)
        if difficulty and difficulty.config:
            cfg = difficulty.config
            level = cfg.get("level", 3)
            hints  = cfg.get("hints", True)
            if level >= 4:
                user_parts.append("Assume expert-level background.")
            elif level <= 2:
                user_parts.append("Use simple language suitable for beginners.")
            if not hints:
                user_parts.append("Do not provide hints or scaffolding.")

        user_context = "\n".join(user_parts)

        # ── 4. Format memory context ─────────────────────────────────
        memory_parts = []
        if memory_context is not None:
            try:
                memories = memory_context.all_memories[:max_memory_nodes]
                if memories:
                    memory_parts.append("Relevant knowledge:")
                    for rm in memories:
                        snippet = rm.node.text[:120].strip()
                        memory_parts.append(f"  • {snippet}")
                if memory_context.principle_nodes:
                    memory_parts.append("Applicable principles:")
                    for p_node in memory_context.principle_nodes[:3]:
                        memory_parts.append(f"  → {p_node.text[:100]}")
                if memory_context.has_contradictions:
                    memory_parts.append(
                        f"⚠ Note: {len(memory_context.contradictions)} "
                        f"conflicting belief(s) detected.")
            except Exception as exc:
                log.debug("PolicyCompiler: memory formatting failed: %s", exc)

        memory_context_str = "\n".join(memory_parts)

        # ── 5. Assemble constraints ───────────────────────────────────
        constraints = []
        if token_budget < 1000:
            constraints.append(f"Keep response under {token_budget // 4} words.")

        # ── 6. Estimate tokens (rough: 1 token ≈ 4 chars) ────────────
        full_text = "\n".join([system_instructions, user_context,
                                memory_context_str, task])
        token_estimate = len(full_text) // 4

        return CompiledPrompt(
            system_instructions=system_instructions,
            user_context=user_context,
            memory_context=memory_context_str,
            task_instructions=task,
            constraints=constraints,
            active_policies=active_policies,
            token_estimate=token_estimate,
            metadata={"user_id": user_id, "n_memories": len(memory_parts)},
        )
