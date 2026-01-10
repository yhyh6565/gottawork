"""Persona agent for character-based interactions."""
from typing import List, Optional, Dict, Any

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import config
from storage.vector_store import VectorKnowledgeBase


PERSONA_PROMPT = """You are {character_name}, a character with the following background and personality:

{character_knowledge}

Character Guidelines:
- Stay in character at all times
- Use speech patterns and mannerisms consistent with your personality
- Reference your background knowledge naturally in conversations
- React to situations based on your established personality traits
- Maintain consistency with your character's relationships and experiences

Current conversation:
{history}

{character_name}: """

CONTENT_GENERATION_PROMPT = """You are {character_name}.
{character_knowledge}

Your task is to compose a {content_type} written by you{recipient_context}.

Topic/Context: {topic}
Tone: {tone}

Instructions:
- Write ONLY the content of the {content_type}.
- Do not include any introductory or concluding remarks outside the content itself.
- Ensure the tone and style matches your personality perfectly.
- Use your characteristic vocabulary and speech patterns.

Content:"""


class CharacterPersona(BaseModel):
    """Character persona configuration."""

    name: str
    personality_traits: List[str] = []
    speech_style: Optional[str] = None
    background_summary: Optional[str] = None
    custom_instructions: Optional[str] = None


class PersonaAgent:
    """Agent that embodies a character persona."""

    def __init__(
        self,
        character_name: str,
        llm_provider: str = "anthropic",
        persona_config: Optional[CharacterPersona] = None,
    ):
        """Initialize the persona agent.

        Args:
            character_name: Name of the character
            llm_provider: "anthropic" or "openai"
            persona_config: Optional custom persona configuration
        """
        self.character_name = character_name
        self.persona_config = persona_config or CharacterPersona(name=character_name)

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

        # Initialize knowledge base
        self.knowledge_base = VectorKnowledgeBase()

        # Load character knowledge
        self.character_knowledge = self._load_character_knowledge()

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=False,
        )

        # Create conversation chain
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=PERSONA_PROMPT,
            partial_variables={
                "character_name": self.character_name,
                "character_knowledge": self.character_knowledge,
            },
        )

        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True,
        )

    def _load_character_knowledge(self) -> str:
        """Load character knowledge from the vector knowledge base.

        Returns:
            Formatted character knowledge string
        """
        # Try to get knowledge from vector store
        knowledge = self.knowledge_base.get_character_knowledge(self.character_name)

        # If no knowledge found, use persona config
        if "No knowledge found" in knowledge:
            knowledge_parts = [f"Character: {self.character_name}"]

            if self.persona_config.background_summary:
                knowledge_parts.append(f"\nBackground: {self.persona_config.background_summary}")

            if self.persona_config.personality_traits:
                traits = ", ".join(self.persona_config.personality_traits)
                knowledge_parts.append(f"\nPersonality: {traits}")

            if self.persona_config.speech_style:
                knowledge_parts.append(f"\nSpeech Style: {self.persona_config.speech_style}")

            if self.persona_config.custom_instructions:
                knowledge_parts.append(f"\n{self.persona_config.custom_instructions}")

            knowledge = "\n".join(knowledge_parts)

        return knowledge

    def chat(self, user_input: str) -> str:
        """Have a conversation with the character.

        Args:
            user_input: User's message

        Returns:
            Character's response
        """
        response = self.conversation.predict(input=user_input)
        return response

    def compose(
        self,
        content_type: str,
        topic: str,
        recipient: Optional[str] = None,
        tone: Optional[str] = "characteristic",
    ) -> str:
        """Compose a specific type of content as the character.

        Args:
            content_type: Type of content (letter, sms, tweet, etc.)
            topic: Topic or context for the content
            recipient: Optional recipient name
            tone: Optional tone instruction

        Returns:
            Generated content
        """
        recipient_context = f" addressed to {recipient}" if recipient else ""
        
        prompt = PromptTemplate(
            input_variables=["content_type", "topic", "tone", "recipient_context"],
            template=CONTENT_GENERATION_PROMPT,
            partial_variables={
                "character_name": self.character_name,
                "character_knowledge": self.character_knowledge,
            },
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            "content_type": content_type,
            "topic": topic,
            "recipient_context": recipient_context,
            "tone": tone,
        })
        
        return response.content if hasattr(response, "content") else str(response)

    def reset_conversation(self) -> None:
        """Reset the conversation memory."""
        self.memory.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of conversation turns
        """
        # This is a simplified version - LangChain memory structure may vary
        history = self.memory.load_memory_variables({})
        return history


def create_persona_agent(
    character_name: str,
    personality_traits: Optional[List[str]] = None,
    speech_style: Optional[str] = None,
    background: Optional[str] = None,
) -> PersonaAgent:
    """Convenience function to create a persona agent.

    Args:
        character_name: Name of the character
        personality_traits: List of personality traits
        speech_style: Description of speech style
        background: Background summary

    Returns:
        PersonaAgent instance
    """
    persona_config = CharacterPersona(
        name=character_name,
        personality_traits=personality_traits or [],
        speech_style=speech_style,
        background_summary=background,
    )

    return PersonaAgent(character_name=character_name, persona_config=persona_config)