"""
Blix v0.3 — Long-Term Memory System
Entry point and Rich-based CLI.

New in v0.3
-----------
* Hierarchical memory: Raw → Session → Daily → Weekly → Project
* Memory importance scoring with configurable weights
* Entity-relationship memory graph
* Dynamic profile evolution (versioned, audited, conflict-resolved)
* Project memory system (first-class project objects)
* Background memory processing (non-blocking chat latency)
* Memory evaluation framework

v0.2 capabilities preserved:
* Chat interface
* Local LLM support (Transformers / Ollama)
* Semantic retrieval using embeddings
* Automatic memory extraction (CoT)
* Structured memory storage
* User profile system

Usage
-----
    python app.py

Commands
--------
    /memory      Last 10 memories with extracted facts
    /profile     User profile (with version info)
    /stats       Full system statistics
    /graph       Memory graph summary
    /projects    List all tracked projects
    /hierarchy   Recent session and weekly summaries
    /eval        Run memory evaluation (if eval dataset exists)
    /session     Flush current session and start a new one
    /help        Command list
    /exit        Quit (flushes session)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from config.settings import settings
from core.background_processor import BackgroundProcessor
from core.embedding_store import EmbeddingStore
from core.hierarchy_manager import HierarchyManager
from core.memory_extractor import MemoryExtractor
from core.memory_graph import MemoryGraph
from core.memory_manager import MemoryManager
from core.memory_retriever import MemoryRetriever
from core.memory_scorer import MemoryScorer, ScoringWeights
from core.profile_evolver import ProfileEvolver
from core.project_manager import ProjectManager
from core.prompt_builder import PromptBuilder
from core.semantic_retriever import SemanticRetriever
from core.tutor_agent import TutorAgent
from llm.provider_factory import build_provider
from utils.helpers import truncate
from utils.logger import get_logger

log = get_logger(__name__)
console = Console()

MEMORY_DIR = Path(__file__).resolve().parent / "memory"


# ---------------------------------------------------------------------------
# Splash
# ---------------------------------------------------------------------------


def _splash() -> None:
    title = Text("BLIX  v0.3  LONG-TERM MEMORY SYSTEM", style="bold cyan", justify="center")
    sub = Text(
        "Hierarchical Memory · Importance Scoring · Memory Graph · Background Processing",
        style="dim", justify="center",
    )
    console.print()
    console.print(Panel.fit(f"{title}\n{sub}", border_style="cyan", padding=(1, 4)))
    console.print()


# ---------------------------------------------------------------------------
# CLI command handlers
# ---------------------------------------------------------------------------


def _cmd_memory(mm: MemoryManager) -> None:
    memories = mm.get_all_memories()
    if not memories:
        console.print("[dim]No memories yet — start chatting![/dim]")
        return
    recent = memories[-10:]
    table = Table(
        title=f"[bold cyan]Last {len(recent)} Memories[/bold cyan]",
        box=box.ROUNDED, show_lines=True, header_style="bold", min_width=70,
    )
    table.add_column("#", style="dim", width=5, no_wrap=True)
    table.add_column("Time", style="cyan", width=12, no_wrap=True)
    table.add_column("Question", style="white", ratio=2)
    table.add_column("Reply preview", style="green", ratio=2)
    table.add_column("Facts / Topics", style="yellow", ratio=2)
    for m in recent:
        facts_str = " · ".join(m.extracted_facts[:2]) if m.extracted_facts else ""
        topics_str = ", ".join(m.topics[:3]) if m.topics else ""
        combined = facts_str or topics_str or "[dim]—[/dim]"
        imp = f" [{m.importance:.2f}]" if m.importance is not None else ""
        table.add_row(
            str(m.id),
            m.timestamp.strftime("%m-%d %H:%M"),
            truncate(m.input, 50),
            truncate(m.output, 60) + imp,
            truncate(combined, 70),
        )
    console.print(table)


def _cmd_profile(agent: TutorAgent, evolver: ProfileEvolver | None) -> None:
    if evolver is not None:
        vp = evolver.versioned
        p = vp.profile
        table = Table(
            title=f"[bold cyan]User Profile (v{vp.version})[/bold cyan]",
            box=box.ROUNDED, show_header=False, padding=(0, 1), min_width=50,
        )
        table.add_column("Field", style="bold cyan", width=14)
        table.add_column("Value", style="white")

        def _val(v: object) -> str:
            if isinstance(v, list):
                return ", ".join(v) if v else "[dim]—[/dim]"
            return str(v) if v else "[dim]—[/dim]"

        table.add_row("Name", _val(p.name))
        table.add_row("Education", _val(p.education))
        table.add_row("Interests", _val(p.interests))
        table.add_row("Projects", _val(p.projects))
        table.add_row("Goals", _val(p.goals))
        table.add_row("Audit entries", str(len(vp.audit)))
        console.print(table)
    else:
        p = agent.memory_manager.profile
        console.print(f"[bold]Profile:[/bold] {p.model_dump()}")


def _cmd_stats(agent: TutorAgent, hierarchy: HierarchyManager | None,
               graph: MemoryGraph | None, pm: ProjectManager | None) -> None:
    mm = agent.memory_manager
    memories = mm.get_all_memories()
    ls = mm.learning_state

    table = Table(
        title="[bold cyan]Blix v0.3 Statistics[/bold cyan]",
        box=box.ROUNDED, show_header=False, padding=(0, 1), min_width=52,
    )
    table.add_column("Metric", style="bold cyan", width=28)
    table.add_column("Value", style="white")

    table.add_row("Total memories stored", str(len(memories)))
    table.add_row("Embedding index size", str(agent.index_size))
    table.add_row("Memories with facts", str(sum(1 for m in memories if m.extracted_facts)))
    table.add_row("Unique topics tracked", str(ls.total_count()))
    if hierarchy:
        table.add_row("Session summaries", str(hierarchy.session_count))
        table.add_row("Daily summaries", str(hierarchy.daily_count))
        table.add_row("Weekly summaries", str(hierarchy.weekly_count))
    if graph:
        table.add_row("Graph nodes", str(graph.node_count))
        table.add_row("Graph edges", str(graph.edge_count))
    if pm:
        table.add_row("Tracked projects", str(pm.count))
    bg = agent.bg_stats
    if bg:
        table.add_row("BG processed", str(bg.get("processed", 0)))
        table.add_row("BG failed", str(bg.get("failed", 0)))
        table.add_row("BG queue size", str(bg.get("queue_size", 0)))
    console.print(table)


def _cmd_graph(graph: MemoryGraph | None) -> None:
    if graph is None:
        console.print("[dim]Memory graph not enabled.[/dim]")
        return
    if graph.node_count == 0:
        console.print("[dim]Graph is empty — start chatting to build it![/dim]")
        return
    table = Table(
        title=f"[bold cyan]Memory Graph ({graph.node_count} nodes, {graph.edge_count} edges)[/bold cyan]",
        box=box.ROUNDED, show_lines=True, header_style="bold",
    )
    table.add_column("From", style="cyan", ratio=1)
    table.add_column("Relation", style="yellow", width=18)
    table.add_column("To", style="green", ratio=1)
    table.add_column("Conf", style="dim", width=6)
    for node in graph.list_nodes()[:15]:
        for rel, target in graph.neighbours(node.id):
            edges = graph.get_edges(from_id=node.id, to_id=target.id)
            conf = f"{edges[0].confidence:.2f}" if edges else "-"
            table.add_row(node.label, rel.value, target.label, conf)
    console.print(table)


def _cmd_projects(pm: ProjectManager | None) -> None:
    if pm is None or pm.count == 0:
        console.print("[dim]No projects tracked yet.[/dim]")
        return
    table = Table(
        title="[bold cyan]Tracked Projects[/bold cyan]",
        box=box.ROUNDED, show_lines=True, header_style="bold",
    )
    table.add_column("Project", style="cyan", ratio=1)
    table.add_column("Status", style="yellow", width=10)
    table.add_column("Goals", style="white", ratio=2)
    table.add_column("Next Actions", style="green", ratio=2)
    for p in pm.list_all():
        table.add_row(
            p.project_name,
            p.current_status,
            truncate(", ".join(p.goals[:2]), 60) or "[dim]—[/dim]",
            truncate(", ".join(p.next_actions[:2]), 60) or "[dim]—[/dim]",
        )
    console.print(table)


def _cmd_hierarchy(hierarchy: HierarchyManager | None) -> None:
    if hierarchy is None:
        console.print("[dim]Hierarchy manager not enabled.[/dim]")
        return
    ctx = hierarchy.get_hierarchy_context(max_sessions=5)
    if ctx:
        console.print(Panel(ctx, title="[bold cyan]Memory Hierarchy[/bold cyan]",
                            border_style="cyan", padding=(0, 1)))
    else:
        console.print("[dim]No summaries yet — finish a session to generate them.[/dim]")


def _cmd_help() -> None:
    table = Table(
        title="[bold cyan]Commands[/bold cyan]",
        box=box.SIMPLE, show_header=False, padding=(0, 2),
    )
    table.add_column("Command", style="bold cyan", width=12)
    table.add_column("Description", style="white")
    commands = [
        ("/memory",    "Show last 10 memories with facts and importance"),
        ("/profile",   "Show versioned user profile with audit count"),
        ("/stats",     "Full system statistics (memory, graph, background, scoring)"),
        ("/graph",     "Memory graph: entities and relationships"),
        ("/projects",  "List all tracked projects"),
        ("/hierarchy", "Recent session and weekly summaries"),
        ("/session",   "Flush current session summary and start a new one"),
        ("/help",      "Show this list"),
        ("/exit",      "Quit (flushes session, stops background worker)"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    console.print(table)


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


def _build_agent() -> tuple[TutorAgent, MemoryManager, HierarchyManager,
                             MemoryGraph, ProjectManager, ProfileEvolver]:
    mem_cfg = settings.memory
    embed_cfg = settings.embed
    llm_cfg = settings.llm

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # v0.2 core
    mm = MemoryManager(
        conversations_file=mem_cfg.conversations_file,
        profile_file=mem_cfg.profile_file,
        learning_state_file=mem_cfg.learning_state_file,
    )
    store = EmbeddingStore(
        embed_model_name=embed_cfg.model,
        embeddings_file=embed_cfg.embeddings_file,
        ids_file=embed_cfg.embedding_ids_file,
        threshold=embed_cfg.threshold,
        top_k=embed_cfg.top_k,
    )
    legacy = MemoryRetriever(
        recent_k=mem_cfg.recent_k,
        fuzzy_top_k=mem_cfg.fuzzy_top_k,
        fuzzy_threshold=mem_cfg.fuzzy_threshold,
        keyword_top_k=mem_cfg.keyword_top_k,
    )
    retriever = SemanticRetriever(
        embedding_store=store,
        legacy_retriever=legacy,
        semantic_top_k=embed_cfg.top_k,
    )
    prompt_builder = PromptBuilder()
    llm = build_provider(llm_cfg)
    extractor = MemoryExtractor(llm=llm, enabled=True) if mem_cfg.auto_extract else None

    # Rebuild index for any unindexed memories
    unindexed = [m for m in mm.get_all_memories() if m.id not in store.indexed_ids]
    if unindexed:
        console.print(f"[dim]Indexing {len(unindexed)} unindexed memor"
                      f"{'y' if len(unindexed) == 1 else 'ies'}…[/dim]")
        retriever.rebuild_index(mm.get_all_memories())

    # v0.3 components
    scorer = MemoryScorer(
        weights=ScoringWeights(
            relevance=0.4, importance=0.3, recency=0.2, frequency=0.1
        )
    )
    hierarchy = HierarchyManager(
        hierarchy_dir=MEMORY_DIR / "hierarchy",
        llm=llm,
    )
    graph = MemoryGraph(graph_file=MEMORY_DIR / "graph.json")
    pm = ProjectManager(projects_file=MEMORY_DIR / "projects.json")
    evolver = ProfileEvolver(versioned_profile_file=MEMORY_DIR / "versioned_profile.json")
    bg = BackgroundProcessor(max_queue_size=100, worker_count=1)

    agent = TutorAgent(
        llm=llm,
        memory_manager=mm,
        retriever=retriever,
        prompt_builder=prompt_builder,
        extractor=extractor,
        scorer=scorer,
        background_processor=bg,
        hierarchy_manager=hierarchy,
        memory_graph=graph,
        project_manager=pm,
        profile_evolver=evolver,
    )
    return agent, mm, hierarchy, graph, pm, evolver


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    _splash()

    try:
        agent, mm, hierarchy, graph, pm, evolver = _build_agent()
    except Exception as exc:
        console.print(f"[bold red]Failed to initialise Blix:[/bold red] {exc}")
        sys.exit(1)

    llm_info = f"{settings.llm.provider}:{settings.llm.model.split('/')[-1]}"
    console.print(
        f"[dim]LLM:[/dim] [cyan]{llm_info}[/cyan]  "
        f"[dim]Embed:[/dim] [cyan]{settings.embed.model}[/cyan]  "
        f"[dim]Memories:[/dim] [cyan]{mm.memory_count()}[/cyan]  "
        f"[dim]User:[/dim] [cyan]{evolver.profile.name or 'unknown'}[/cyan]  "
        f"[dim]Sessions:[/dim] [cyan]{hierarchy.session_count}[/cyan]  "
        f"[dim]Graph:[/dim] [cyan]{graph.node_count}N/{graph.edge_count}E[/cyan]"
    )
    console.print("[dim]Type [bold]/help[/bold] for commands.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Flushing session…[/dim]")
            agent.shutdown()
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower == "/exit":
            console.print("[dim]Flushing session…[/dim]")
            agent.shutdown()
            console.print("[dim]Goodbye![/dim]")
            break
        elif lower == "/memory":
            _cmd_memory(mm)
        elif lower == "/profile":
            _cmd_profile(agent, evolver)
        elif lower == "/stats":
            _cmd_stats(agent, hierarchy, graph, pm)
        elif lower == "/graph":
            _cmd_graph(graph)
        elif lower == "/projects":
            _cmd_projects(pm)
        elif lower == "/hierarchy":
            _cmd_hierarchy(hierarchy)
        elif lower == "/session":
            agent.new_session()
            console.print("[dim]Session flushed. New session started.[/dim]")
        elif lower == "/help":
            _cmd_help()
        elif lower.startswith("/"):
            console.print(
                f"[yellow]Unknown command[/yellow] [bold]{user_input!r}[/bold]. "
                "Type [bold]/help[/bold]."
            )
        else:
            console.print()
            with console.status("[cyan]Blix is thinking…[/cyan]", spinner="dots"):
                try:
                    response = agent.chat(user_input)
                except RuntimeError as exc:
                    console.print(f"[bold red]Error:[/bold red] {exc}")
                    continue

            console.print(Panel(
                Markdown(response),
                title="[bold cyan]Blix[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            ))
            console.print()


if __name__ == "__main__":
    main()
