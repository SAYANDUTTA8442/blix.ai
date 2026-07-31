"""
Tool Registry & Tool Implementations — Blix v0.3.5  (Modules 4 & 5)

Defines the abstract ``Tool`` interface and all built-in tools:

    MemorySearchTool     — semantic search over Blix memory
    MemoryWriteTool      — persist a fact into memory
    WebSearchTool        — web search via DuckDuckGo (no API key)
    FileTool             — read/write local files
    PythonTool           — execute Python snippets in a sandbox
    SynthesisTool        — run KnowledgeSynthesisEngine on gathered context
    ReasoningTool        — CognitiveQueryEngine graph query
    DocumentTool         — process a document via DocumentProcessor

``ToolRegistry`` manages tool discovery, schema validation, and the
Tool Selection Engine (Module 5) that picks the best tool for a task.

Python 3.10 compatible.
Sensitive tools (PythonTool, FileTool) have configurable guards.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.types import ExecutionResult, ExecutionStatus, Task
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Tool(ABC):
    """
    Abstract base for all Blix agent tools.

    Subclasses implement ``execute()`` and optionally ``can_handle()``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description for the Tool Selection Engine."""
        ...

    @property
    def requires_confirmation(self) -> bool:
        """If True, executor pauses and asks human approval before running."""
        return False

    def can_handle(self, task: Task) -> float:
        """
        Return a confidence score 0–1 that this tool is appropriate for
        the given task.  Default: keyword match on description.
        """
        task_text = (task.title + " " + task.description).lower()
        keywords = re.findall(r"[a-z]+", self.description.lower())
        hits = sum(1 for k in keywords if k in task_text and len(k) > 3)
        return min(1.0, hits / max(1, len(keywords) * 0.3))

    @abstractmethod
    def execute(self, task: Task, context: dict) -> ExecutionResult:
        """
        Execute the tool for the given task.

        Parameters
        ----------
        task:
            The task to execute.
        context:
            Working memory context (key → value).

        Returns
        -------
        ExecutionResult
        """
        ...

    def _result(
        self,
        task: Task,
        status: ExecutionStatus,
        output: str,
        error: str = "",
        raw: Any = None,
        duration_ms: float = 0.0,
    ) -> ExecutionResult:
        return ExecutionResult(
            task_id=task.task_id,
            tool_name=self.name,
            status=status,
            output=output,
            error=error,
            raw=raw,
            duration_ms=duration_ms,
        )


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


class MemorySearchTool(Tool):
    """Semantic search over Blix long-term memory."""

    def __init__(self, memory_manager: object, retriever: object) -> None:
        self._mm = memory_manager
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return "Search long-term memory for relevant past conversations and facts."

    def can_handle(self, task: Task) -> float:
        text = (task.title + " " + task.description).lower()
        keywords = {"recall", "remember", "past", "memory", "history", "previous", "stored"}
        return 0.8 if any(k in text for k in keywords) else 0.2

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        query = task.description or task.title
        try:
            all_memories = self._mm.get_all_memories()  # type: ignore[union-attr]
            results = self._retriever.retrieve(all_memories, query)[:5]  # type: ignore[union-attr]
            if not results:
                return self._result(task, ExecutionStatus.SUCCESS,
                                    "No relevant memories found.", duration_ms=_ms(t0))
            lines = [f"[{m.id}] {m.output[:200]}" for m in results]
            output = "Memory search results:\n" + "\n".join(lines)
            return self._result(task, ExecutionStatus.SUCCESS, output,
                                raw=results, duration_ms=_ms(t0))
        except Exception as exc:
            return self._result(task, ExecutionStatus.ERROR, "", str(exc), duration_ms=_ms(t0))


class MemoryWriteTool(Tool):
    """Write a fact/observation directly into Blix long-term memory."""

    def __init__(self, memory_manager: object) -> None:
        self._mm = memory_manager

    @property
    def name(self) -> str:
        return "memory_write"

    @property
    def description(self) -> str:
        return "Save an important fact or observation to long-term memory."

    def can_handle(self, task: Task) -> float:
        text = (task.title + " " + task.description).lower()
        return 0.8 if any(k in text for k in ("save", "store", "record", "write", "remember")) else 0.1

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        content = task.metadata.get("content") or task.description
        try:
            entry = self._mm.add_memory(  # type: ignore[union-attr]
                f"[Agent] {task.title}", content
            )
            output = f"Saved to memory as entry #{entry.id}."
            return self._result(task, ExecutionStatus.SUCCESS, output, duration_ms=_ms(t0))
        except Exception as exc:
            return self._result(task, ExecutionStatus.ERROR, "", str(exc), duration_ms=_ms(t0))


# ---------------------------------------------------------------------------
# Web search tool (DuckDuckGo Instant Answer API — no key required)
# ---------------------------------------------------------------------------


class WebSearchTool(Tool):
    """Web search using DuckDuckGo Instant Answer API (no API key required)."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information, recent papers, news, or documentation."

    def can_handle(self, task: Task) -> float:
        text = (task.title + " " + task.description).lower()
        keywords = {"search", "web", "latest", "recent", "current", "news", "paper",
                    "online", "find", "lookup", "research"}
        return 0.85 if any(k in text for k in keywords) else 0.15

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        query = task.metadata.get("query") or task.title
        try:
            import urllib.request
            import urllib.parse
            url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query) + "&format=json&no_html=1"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            related = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:3] if isinstance(r, dict)]
            if abstract:
                output = f"Web search result for '{query}':\n{abstract}"
                if related:
                    output += "\n\nRelated:\n" + "\n".join(f"- {r}" for r in related if r)
            elif related:
                output = f"Web search for '{query}':\n" + "\n".join(f"- {r}" for r in related if r)
            else:
                output = f"No direct results found for '{query}'. Try a more specific query."
            return self._result(task, ExecutionStatus.SUCCESS, output,
                                raw=data, duration_ms=_ms(t0))
        except Exception as exc:
            return self._result(task, ExecutionStatus.FAILURE,
                                f"Web search failed: {exc}",
                                str(exc), duration_ms=_ms(t0))


# ---------------------------------------------------------------------------
# File tool
# ---------------------------------------------------------------------------


class FileTool(Tool):
    """Read or write files in a designated workspace directory."""

    def __init__(self, workspace_dir: Path, allow_write: bool = True) -> None:
        self._workspace = workspace_dir
        self._allow_write = allow_write
        workspace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "file_tool"

    @property
    def description(self) -> str:
        return "Read or write files: load documents, save results, create notes."

    @property
    def requires_confirmation(self) -> bool:
        return False  # Write within workspace only — no confirmation needed

    def can_handle(self, task: Task) -> float:
        text = (task.title + " " + task.description).lower()
        return 0.85 if any(k in text for k in ("file", "read", "write", "save", "load", "document")) else 0.1

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        op = task.metadata.get("op", "read")
        filename = task.metadata.get("filename", "output.txt")
        path = self._workspace / filename

        # Guard: stay within workspace
        try:
            path.resolve().relative_to(self._workspace.resolve())
        except ValueError:
            return self._result(task, ExecutionStatus.ERROR,
                                "", "Path escape attempt blocked.", duration_ms=_ms(t0))

        if op == "read":
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                return self._result(task, ExecutionStatus.SUCCESS,
                                    f"File '{filename}' ({len(content)} chars):\n{content[:2000]}",
                                    raw=content, duration_ms=_ms(t0))
            except FileNotFoundError:
                return self._result(task, ExecutionStatus.FAILURE,
                                    f"File '{filename}' not found.", duration_ms=_ms(t0))
        elif op == "write" and self._allow_write:
            content = task.metadata.get("content", task.description)
            path.write_text(content, encoding="utf-8")
            return self._result(task, ExecutionStatus.SUCCESS,
                                f"Wrote {len(content)} chars to '{filename}'.",
                                duration_ms=_ms(t0))
        elif op == "list":
            files = [f.name for f in self._workspace.iterdir() if f.is_file()]
            return self._result(task, ExecutionStatus.SUCCESS,
                                "Workspace files:\n" + "\n".join(files), duration_ms=_ms(t0))
        else:
            return self._result(task, ExecutionStatus.ERROR,
                                "", f"Unknown op '{op}'.", duration_ms=_ms(t0))


# ---------------------------------------------------------------------------
# Python execution tool (sandboxed)
# ---------------------------------------------------------------------------


class PythonTool(Tool):
    """
    Execute Python code snippets safely.

    Sandbox: only ``builtins`` + safe stdlib modules whitelisted.
    No network, no filesystem, no subprocess from within the snippet.
    """

    _SAFE_BUILTINS = {
        "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
        "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
        "pow", "print", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    }
    _TIMEOUT_SECS = 5.0

    @property
    def name(self) -> str:
        return "python_tool"

    @property
    def description(self) -> str:
        return "Execute Python code for data analysis, calculations, or processing."

    def can_handle(self, task: Task) -> float:
        text = (task.title + " " + task.description).lower()
        return 0.9 if any(k in text for k in ("python", "code", "compute", "calculate",
                                               "analyse", "analyze", "plot", "script")) else 0.1

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        code = task.metadata.get("code") or _extract_code_block(task.description)
        if not code:
            return self._result(task, ExecutionStatus.ERROR, "",
                                "No code provided. Add 'code' to task.metadata.", duration_ms=_ms(t0))

        output_lines: list[str] = []
        safe_globals = {
            "__builtins__": {k: __builtins__[k] for k in self._SAFE_BUILTINS  # type: ignore[index]
                             if k in __builtins__},  # type: ignore[operator]
            "context": context,
        }
        # Allow math and json
        import math as _math
        safe_globals["math"] = _math
        safe_globals["json"] = json

        # Capture print output
        import io, contextlib
        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code, "<agent_code>", "exec"), safe_globals)
            output = stdout_capture.getvalue().strip() or "(no output)"
            return self._result(task, ExecutionStatus.SUCCESS, output,
                                duration_ms=_ms(t0))
        except Exception as exc:
            return self._result(task, ExecutionStatus.FAILURE,
                                stdout_capture.getvalue(),
                                f"{type(exc).__name__}: {exc}", duration_ms=_ms(t0))


# ---------------------------------------------------------------------------
# Synthesis tool (wraps KnowledgeSynthesisEngine)
# ---------------------------------------------------------------------------


class SynthesisTool(Tool):
    """Synthesise a knowledge report from gathered context in working memory."""

    def __init__(self, synthesis_engine: object) -> None:
        self._synthesis = synthesis_engine

    @property
    def name(self) -> str:
        return "synthesis"

    @property
    def description(self) -> str:
        return "Synthesise findings into a structured knowledge report."

    def can_handle(self, task: Task) -> float:
        text = (task.title + " " + task.description).lower()
        return 0.85 if any(k in text for k in ("synthesize", "synthesise", "summarize",
                                                "report", "combine", "integrate")) else 0.2

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        from knowledge.synthesis import SynthesisSource
        sources = []
        for key, val in context.items():
            if isinstance(val, str) and len(val) > 20:
                sources.append(SynthesisSource(kind="working_memory", ref_id=key, text=val[:600]))
        if not sources:
            return self._result(task, ExecutionStatus.FAILURE,
                                "No context to synthesise.", duration_ms=_ms(t0))
        try:
            report = self._synthesis.synthesize(sources)  # type: ignore[union-attr]
            output = f"Synthesis Report: {report.title}\n\n{report.narrative}"
            if report.key_points:
                output += "\n\nKey points:\n" + "\n".join(f"- {p}" for p in report.key_points)
            return self._result(task, ExecutionStatus.SUCCESS, output,
                                raw=report.to_dict(), duration_ms=_ms(t0))
        except Exception as exc:
            return self._result(task, ExecutionStatus.ERROR, "", str(exc), duration_ms=_ms(t0))


# ---------------------------------------------------------------------------
# Reasoning tool (wraps CognitiveQueryEngine)
# ---------------------------------------------------------------------------


class ReasoningTool(Tool):
    """Query the knowledge graph using the CognitiveQueryEngine."""

    def __init__(self, cognitive_query_engine: object) -> None:
        self._cqe = cognitive_query_engine

    @property
    def name(self) -> str:
        return "reasoning"

    @property
    def description(self) -> str:
        return "Reason over the knowledge graph to answer questions about entities and relationships."

    def can_handle(self, task: Task) -> float:
        text = (task.title + " " + task.description).lower()
        return 0.9 if any(k in text for k in ("graph", "relationship", "reasoning",
                                               "who", "what does", "infer", "connect")) else 0.15

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        query = task.metadata.get("query") or task.description
        try:
            result = self._cqe.query(query)  # type: ignore[union-attr]
            if result.is_empty():
                output = f"Graph query '{query}': no results found."
            else:
                output = f"Graph query '{query}':\n→ {', '.join(result.answer)}"
                output += f"\n\nExplanation: {result.trace.explanation}"
            return self._result(task, ExecutionStatus.SUCCESS, output,
                                raw=result.to_dict(), duration_ms=_ms(t0))
        except Exception as exc:
            return self._result(task, ExecutionStatus.ERROR, "", str(exc), duration_ms=_ms(t0))


# ---------------------------------------------------------------------------
# LLM tool (generate text for a subtask using the chat LLM)
# ---------------------------------------------------------------------------


class LLMTool(Tool):
    """Generate text for a subtask using the Blix LLM (e.g. draft, summarize, explain)."""

    def __init__(self, llm: object) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "llm"

    @property
    def description(self) -> str:
        return "Generate text: draft, explain, summarise, or answer questions using the LLM."

    def can_handle(self, task: Task) -> float:
        return 0.5  # generic fallback with medium confidence

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        t0 = time.monotonic()
        prompt = task.metadata.get("prompt") or task.description
        # Inject working memory context
        if context:
            ctx_text = "\n".join(f"{k}: {str(v)[:200]}" for k, v in list(context.items())[:5])
            prompt = f"Context:\n{ctx_text}\n\nTask: {prompt}"
        try:
            output = self._llm.generate(prompt).strip()  # type: ignore[union-attr]
            return self._result(task, ExecutionStatus.SUCCESS, output, duration_ms=_ms(t0))
        except Exception as exc:
            return self._result(task, ExecutionStatus.ERROR, "", str(exc), duration_ms=_ms(t0))


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


@dataclass
class ToolMatch:
    """A tool scored for a specific task."""

    tool: Tool
    score: float

    def __str__(self) -> str:
        return f"{self.tool.name} (score={self.score:.2f})"


class ToolRegistry:
    """
    Manages all registered tools and implements the Tool Selection Engine
    (Module 5): picks the best tool for a given task.

    Parameters
    ----------
    tools:
        Initial list of tools to register.
    """

    def __init__(self, tools: Optional[list[Tool]] = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in (tools or []):
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        log.debug("ToolRegistry: registered '%s'", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schema(self) -> list[dict]:
        """Return OpenAI-style tool schemas for all registered tools."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    # ------------------------------------------------------------------
    # Tool Selection Engine (Module 5)
    # ------------------------------------------------------------------

    def select_tool(self, task: Task) -> Optional[Tool]:
        """
        Select the best tool for a task.

        Priority:
        1. If task.tool_hint is set and the named tool is registered → use it.
        2. Otherwise rank all tools by ``can_handle()`` score → pick highest.
        3. Return None if no tool scores above 0.1.
        """
        # Explicit hint takes priority
        if task.tool_hint and task.tool_hint in self._tools:
            return self._tools[task.tool_hint]

        # Rank by can_handle score
        scored = [ToolMatch(t, t.can_handle(task)) for t in self._tools.values()]
        scored.sort(key=lambda m: -m.score)
        best = scored[0] if scored else None
        if best and best.score > 0.1:
            log.debug("ToolRegistry: selected '%s' (score=%.2f) for task '%s'",
                      best.tool.name, best.score, task.title)
            return best.tool
        return None

    def rank_tools(self, task: Task) -> list[ToolMatch]:
        """Return all tools ranked by suitability for the task."""
        scored = [ToolMatch(t, t.can_handle(task)) for t in self._tools.values()]
        return sorted(scored, key=lambda m: -m.score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 1)


def _extract_code_block(text: str) -> str:
    """Extract a ```python ... ``` block from text, or return text itself."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: return text if it looks like code
    if any(kw in text for kw in ("def ", "import ", "print(", "for ", "if ")):
        return text.strip()
    return ""
