"""Research agent for gathering information about characters and works."""
from typing import List, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from config import config
from storage.vector_store import VectorKnowledgeBase, CharacterKnowledge
from storage.markdown_exporter import (
    MarkdownExporter,
    ResearchReport,
    ResearchSection,
)
from tools.google_search import create_google_search_tool
from tools.twitter_search import create_twitter_search_tool
from tools.community_scraper import create_community_search_tool
from tools.web_scraper import create_web_scraper_tool


RESEARCH_PROMPT = """You are an expert research agent specializing in entertainment content, characters, and web novels.

Your task is to thoroughly research the given subject using the available tools and compile a comprehensive report.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Research Guidelines:
1. Start with community searches (Namu Wiki) for structured information
2. Use Google search for broader context and recent updates
3. Check Twitter for fan reactions and trending discussions
4. Scrape relevant web pages for detailed information
5. Synthesize all findings into a coherent narrative

When researching a CHARACTER:
- Background and profile
- Appearance and personality traits
- Role in the story
- Relationships with other characters
- Character development and arc
- Fan reception and popular interpretations

When researching a WORK (web novel, manga, etc.):
- Plot summary and genre
- Main characters
- Publication history
- Popularity and reception
- Themes and motifs
- Fan community and discussions

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to compile the report
Final Answer: A comprehensive summary of all findings

Current research subject: {input}

{agent_scratchpad}
"""


class ResearchAgent:
    """Agent for researching characters and entertainment content."""

    def __init__(self, llm_provider: str = "anthropic"):
        """Initialize the research agent.

        Args:
            llm_provider: "anthropic" or "openai"
        """
        # Initialize LLM
        if llm_provider == "anthropic":
            if not config.llm.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in .env file")
            self.llm = ChatAnthropic(
                model=config.llm.default_model,
                api_key=config.llm.anthropic_api_key,
                temperature=config.llm.temperature,
            )
        else:
            if not config.llm.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in .env file")
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                api_key=config.llm.openai_api_key,
                temperature=config.llm.temperature,
            )

        # Initialize tools
        self.tools = self._create_tools()

        # Initialize vector knowledge base
        self.knowledge_base = VectorKnowledgeBase()

        # Initialize markdown exporter
        self.exporter = MarkdownExporter()

        # Create agent
        prompt = PromptTemplate.from_template(RESEARCH_PROMPT)
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
        )

    def _create_tools(self) -> List[Tool]:
        """Create the tools for the agent.

        Returns:
            List of LangChain tools
        """
        tools = []

        # Add community search
        tools.append(create_community_search_tool())

        # Add Google search
        try:
            tools.append(create_google_search_tool())
        except ImportError:
            print("Google search not available - install googlesearch-python")

        # Add Twitter search
        try:
            tools.append(create_twitter_search_tool())
        except (ImportError, ValueError) as e:
            print(f"Twitter search not available: {e}")

        # Add web scraper
        tools.append(create_web_scraper_tool())

        return tools

    def research(
        self,
        subject: str,
        subject_type: str = "character",
        save_to_kb: bool = True,
    ) -> ResearchReport:
        """Conduct research on a subject.

        Args:
            subject: The subject to research (character name, work title, etc.)
            subject_type: Type of subject ("character", "work", etc.)
            save_to_kb: Whether to save findings to the knowledge base

        Returns:
            ResearchReport with all findings
        """
        print(f"\n🔍 Starting research on: {subject} ({subject_type})")
        print("=" * 60)

        # Run the agent
        result = self.agent_executor.invoke({"input": subject})

        # Extract the findings
        findings = result.get("output", "No findings available")

        print("\n✅ Research completed!")
        print("=" * 60)

        # Create structured report using LLM
        report = self._create_structured_report(subject, subject_type, findings)

        # Save to knowledge base
        if save_to_kb:
            self._save_to_knowledge_base(subject, findings)

        # Export to markdown
        filepath = self.exporter.export(report)
        print(f"\n📄 Report saved to: {filepath}")

        return report

    def _create_structured_report(
        self,
        subject: str,
        subject_type: str,
        findings: str,
    ) -> ResearchReport:
        """Create a structured report from raw findings.

        Args:
            subject: The research subject
            subject_type: Type of subject
            findings: Raw research findings

        Returns:
            Structured ResearchReport
        """
        # Use LLM to structure the findings
        structuring_prompt = f"""Given the following research findings about {subject}, create a structured summary.

Research findings:
{findings}

Please provide:
1. A brief summary (2-3 sentences)
2. Key sections with detailed information

Format your response as JSON with this structure:
{{
    "summary": "brief summary here",
    "sections": [
        {{"title": "Section Title", "content": "Section content"}},
        ...
    ]
}}
"""

        try:
            response = self.llm.invoke(structuring_prompt)
            # Parse the response (simplified - in production, use proper JSON parsing)
            summary = findings[:500]  # Fallback
            sections = [
                ResearchSection(title="Research Findings", content=findings)
            ]
        except Exception as e:
            print(f"Error structuring report: {e}")
            summary = f"Research findings for {subject}"
            sections = [
                ResearchSection(title="Research Findings", content=findings)
            ]

        return ResearchReport(
            subject=subject,
            subject_type=subject_type,
            summary=summary,
            sections=sections,
            sources=self._extract_sources(findings),
            created_at=datetime.now(),
        )

    def _save_to_knowledge_base(self, subject: str, findings: str) -> None:
        """Save research findings to the vector knowledge base.

        Args:
            subject: The research subject
            findings: Research findings to save
        """
        knowledge = CharacterKnowledge(
            character_name=subject,
            source="research_agent",
            content=findings,
            metadata={"timestamp": datetime.now().isoformat()},
        )

        self.knowledge_base.add_knowledge(knowledge)
        self.knowledge_base.persist()
        print(f"💾 Saved to knowledge base: {subject}")

    def _extract_sources(self, findings: str) -> List[str]:
        """Extract source URLs from findings.

        Args:
            findings: Research findings text

        Returns:
            List of source URLs
        """
        sources = []
        # Simple URL extraction (could be improved with regex)
        lines = findings.split("\n")
        for line in lines:
            if "http://" in line or "https://" in line:
                # Extract URL (simplified)
                words = line.split()
                for word in words:
                    if word.startswith("http"):
                        sources.append(word.rstrip(".,;)"))

        return list(set(sources))  # Remove duplicates
