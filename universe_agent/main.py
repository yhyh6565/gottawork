#!/usr/bin/env python3
"""Main CLI entry point for the research agent."""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.researcher import ResearchAgent
from agents.persona import create_persona_agent
from storage.vector_store import VectorKnowledgeBase

app = typer.Typer(
    name="research",
    help="AI-powered research agent for character and content analysis",
)
console = Console()


@app.command()
def research(
    subject: str = typer.Argument(..., help="Subject to research (character name or work title)"),
    subject_type: str = typer.Option(
        "character",
        "--type",
        "-t",
        help="Type of subject: character, work, etc.",
    ),
    llm: str = typer.Option(
        "anthropic",
        "--llm",
        "-l",
        help="LLM provider: anthropic or openai",
    ),
    no_save: bool = typer.Option(
        False,
        "--no-save",
        help="Don't save findings to knowledge base",
    ),
):
    """Research a character or work and generate a report."""
    console.print(
        Panel.fit(
            f"[bold cyan]Research Agent[/bold cyan]\n"
            f"Subject: [yellow]{subject}[/yellow]\n"
            f"Type: [green]{subject_type}[/green]",
            border_style="cyan",
        )
    )

    try:
        # Initialize agent
        with console.status("[bold green]Initializing research agent..."):
            agent = ResearchAgent(llm_provider=llm)

        # Conduct research
        report = agent.research(
            subject=subject,
            subject_type=subject_type,
            save_to_kb=not no_save,
        )

        console.print("\n[bold green]✓[/bold green] Research completed successfully!")
        console.print(f"[dim]Report saved to: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def compose(
    character: str = typer.Argument(..., help="Character name"),
    content_type: str = typer.Option(..., "--type", "-t", help="Type of content (e.g. letter, sms, tweet)"),
    topic: str = typer.Option(..., "--topic", "-T", help="Topic or context for the content"),
    recipient: Optional[str] = typer.Option(None, "--recipient", "-to", help="Recipient name"),
    tone: str = typer.Option("characteristic", "--tone", help="Tone instruction"),
    llm: str = typer.Option(
        "anthropic",
        "--llm",
        "-l",
        help="LLM provider: anthropic or openai",
    ),
):
    """Generate content written by a character persona."""
    console.print(
        Panel.fit(
            f"[bold magenta]Character Composer[/bold magenta]\n"
            f"Character: [yellow]{character}[/yellow]\n"
            f"Type: [cyan]{content_type}[/cyan]\n"
            f"Topic: [green]{topic}[/green]",
            border_style="magenta",
        )
    )

    try:
        # Initialize persona agent
        with console.status(f"[bold green]Loading {character}'s persona..."):
            agent = create_persona_agent(character_name=character)

        # Generate content
        with console.status("[dim]Composing...[/dim]"):
            content = agent.compose(
                content_type=content_type,
                topic=topic,
                recipient=recipient,
                tone=tone,
            )

        console.print("\n[bold green]✓[/bold green] Generated Content:\n")
        console.print(Panel(content, title=f"{character}'s {content_type}", border_style="cyan"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def kb_search(
    query: str = typer.Argument(..., help="Search query"),
    character: Optional[str] = typer.Option(
        None,
        "--character",
        "-c",
        help="Filter by character name",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of results to return",
    ),
):
    """Search the knowledge base."""
    console.print(
        Panel.fit(
            f"[bold blue]Knowledge Base Search[/bold blue]\n"
            f"Query: [yellow]{query}[/yellow]"
            + (f"\nCharacter: [green]{character}[/green]" if character else ""),
            border_style="blue",
        )
    )

    try:
        kb = VectorKnowledgeBase()
        results = kb.search(query=query, character_name=character, k=limit)

        if not results:
            console.print("\n[yellow]No results found.[/yellow]")
            return

        console.print(f"\n[bold]Found {len(results)} results:[/bold]\n")

        for i, result in enumerate(results, 1):
            console.print(f"[bold cyan]{i}. [dim](Score: {result['relevance_score']:.3f})[/dim][/bold cyan]")
            console.print(f"[dim]Source:[/dim] {result['metadata'].get('source', 'unknown')}")
            console.print(result["content"][:200] + "...")
            console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def kb_info(
    character: str = typer.Argument(..., help="Character name"),
):
    """Display all knowledge about a character."""
    console.print(
        Panel.fit(
            f"[bold blue]Character Knowledge[/bold blue]\n"
            f"Character: [yellow]{character}[/yellow]",
            border_style="blue",
        )
    )

    try:
        kb = VectorKnowledgeBase()
        knowledge = kb.get_character_knowledge(character)

        console.print("\n")
        console.print(Panel(knowledge, title=f"Knowledge: {character}", border_style="blue"))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def init():
    """Initialize the project (create .env from template)."""
    env_example = Path(".env.example")
    env_file = Path(".env")

    if env_file.exists():
        if not typer.confirm(".env already exists. Overwrite?"):
            console.print("[yellow]Initialization cancelled.[/yellow]")
            raise typer.Exit(0)

    if not env_example.exists():
        console.print("[red].env.example not found![/red]")
        raise typer.Exit(1)

    # Copy .env.example to .env
    env_file.write_text(env_example.read_text())

    console.print(
        Panel.fit(
            "[bold green]✓ Initialization complete![/bold green]\n\n"
            "[yellow]Next steps:[/yellow]\n"
            "1. Edit .env and add your API keys\n"
            "2. Install dependencies: pip install -r requirements.txt\n"
            "3. Run: python main.py research 'character name'",
            border_style="green",
        )
    )


@app.command()
def version():
    """Show version information."""
    console.print(
        Panel.fit(
            "[bold cyan]Research Agent[/bold cyan]\n"
            "Version: [yellow]0.1.0[/yellow]\n"
            "LangChain-powered character research tool",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    app()
