"""
Prompt template module for the Technical RAG Assistant.

This module provides prompt templates for RAG-based question answering:
- System instructions
- Query formatting
- Context integration
- Citation formatting
- Response structuring

Optimized for technical documentation and knowledge retrieval.
"""

import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class MedicalPromptTemplate:
    """
    Manages prompt templates for technical RAG queries.
    
    Note: Despite the class name (for backwards compatibility with main.py),
    this is now optimized for general technical content, not medical content.
    """
    
    def __init__(
        self,
        include_citations: bool = True,
        include_disclaimer: bool = False,
        citation_format: str = "inline"
    ):
        """
        Initialize the prompt template.
        
        Args:
            include_citations: Whether to request citations in responses
            include_disclaimer: Whether to add a disclaimer (set to False for technical content)
            citation_format: Format for citations ('inline', 'endnotes', 'none')
        """
        self.include_citations = include_citations
        self.include_disclaimer = include_disclaimer
        self.citation_format = citation_format
        
        logger.info(f"PromptTemplate initialized: citations={include_citations}, format={citation_format}")
    
    def get_system_instruction(self) -> str:
        """
        Get the system instruction for the LLM.
        
        Returns:
            System instruction string
        """
        instruction = """You are a helpful technical assistant that answers questions based on provided documentation and context.

Your responsibilities:
1. Provide accurate, clear, and concise answers based on the given context
2. Stay strictly within the information provided in the context
3. If the context doesn't contain enough information to answer the question, clearly state this
4. Use technical terminology appropriately and explain complex concepts when needed
5. Structure your answers logically with clear explanations"""

        if self.include_citations:
            instruction += "\n6. Reference the documents you use by their document numbers (e.g., 'According to Document 1...')"
        
        if self.include_disclaimer:
            instruction += "\n7. Include appropriate disclaimers when discussing sensitive topics"
        
        instruction += "\n\nAlways prioritize accuracy over completeness. It's better to say 'I don't know' than to speculate beyond the provided context."
        
        return instruction
    
    def format_rag_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format a complete RAG prompt with query and context.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            conversation_history: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append("Previous Conversation:")
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")
        
        # Add context
        prompt_parts.append("Context from Documentation:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add instructions
        instructions = []
        instructions.append("Based on the context above, please answer the following question:")
        
        if self.include_citations:
            if self.citation_format == "inline":
                instructions.append("Include inline citations by referencing document numbers (e.g., '[Document 1]').")
            elif self.citation_format == "endnotes":
                instructions.append("Add numbered citations and provide a references section at the end.")
        
        instructions.append("If the context doesn't contain sufficient information, clearly state what's missing.")
        
        prompt_parts.append(" ".join(instructions))
        prompt_parts.append("")
        
        # Add question
        prompt_parts.append(f"Question: {query}")
        
        return "\n".join(prompt_parts)
    
    def format_simple_prompt(self, query: str, context: str) -> str:
        """
        Format a simple prompt without extra instructions.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Simple formatted prompt
        """
        return f"""Context:
{context}

Question: {query}

Answer:"""
    
    def format_conversational_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Format a prompt for conversational interaction.
        
        Args:
            query: Current user query
            context: Retrieved context
            conversation_history: Previous messages
            
        Returns:
            Conversational prompt
        """
        prompt_parts = []
        
        # Add conversation history
        if conversation_history:
            prompt_parts.append("Conversation History:")
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get('role', 'user').capitalize()
                content = msg.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")
        
        # Add context
        prompt_parts.append("Relevant Documentation:")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Add current query
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("")
        prompt_parts.append("Assistant: ")
        
        return "\n".join(prompt_parts)
    
    def format_followup_prompt(
        self,
        original_query: str,
        original_answer: str,
        followup_query: str,
        new_context: str
    ) -> str:
        """
        Format a prompt for follow-up questions.
        
        Args:
            original_query: Previous question
            original_answer: Previous answer
            followup_query: New follow-up question
            new_context: Newly retrieved context
            
        Returns:
            Follow-up prompt
        """
        return f"""Previous Question: {original_query}

Previous Answer: {original_answer}

Additional Context:
{new_context}

Follow-up Question: {followup_query}

Please provide a comprehensive answer that builds on the previous conversation and incorporates the new context."""
    
    def format_extraction_prompt(self, query: str, context: str) -> str:
        """
        Format a prompt for extracting specific information.
        
        Args:
            query: What to extract
            context: Context to extract from
            
        Returns:
            Extraction prompt
        """
        return f"""Context:
{context}

Task: Extract the following information from the context above:
{query}

Instructions:
- Only include information explicitly stated in the context
- If information is not found, state "Not found in context"
- Be precise and concise
- Cite document numbers for each piece of information

Extracted Information:"""
    
    def format_summary_prompt(self, context: str, focus: Optional[str] = None) -> str:
        """
        Format a prompt for summarizing content.
        
        Args:
            context: Content to summarize
            focus: Optional focus area for the summary
            
        Returns:
            Summary prompt
        """
        prompt = f"""Context:
{context}

Task: Provide a comprehensive summary of the above content."""
        
        if focus:
            prompt += f"\n\nFocus specifically on: {focus}"
        
        prompt += """

Summary Guidelines:
- Capture key points and main ideas
- Maintain technical accuracy
- Keep it concise but informative
- Use clear, structured formatting

Summary:"""
        
        return prompt
    
    def format_comparison_prompt(
        self,
        query: str,
        context1: str,
        context2: str,
        label1: str = "Option 1",
        label2: str = "Option 2"
    ) -> str:
        """
        Format a prompt for comparing two pieces of information.
        
        Args:
            query: Comparison question
            context1: First context
            context2: Second context
            label1: Label for first context
            label2: Label for second context
            
        Returns:
            Comparison prompt
        """
        return f"""{label1}:
{context1}

{label2}:
{context2}

Question: {query}

Please provide a detailed comparison addressing:
- Key similarities
- Key differences
- Advantages and disadvantages of each
- Recommendations based on the comparison

Comparison:"""
    
    def add_disclaimer(self, response: str) -> str:
        """
        Add a disclaimer to a response if enabled.
        
        Args:
            response: Original response
            
        Returns:
            Response with disclaimer if enabled
        """
        if not self.include_disclaimer:
            return response
        
        disclaimer = "\n\n---\nNote: This information is for reference purposes only. Always verify critical information from authoritative sources."
        
        return response + disclaimer


if __name__ == "__main__":
    # Test the prompt templates
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("PROMPT TEMPLATE TEST")
    print("="*80)
    
    # Initialize template
    template = MedicalPromptTemplate(
        include_citations=True,
        citation_format="inline"
    )
    
    # Test data
    test_query = "What is machine learning and how does it work?"
    test_context = """[Document 1]
Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms that can identify patterns and make decisions based on input data.

[Document 2]
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each type uses different approaches to train models and solve different kinds of problems."""
    
    # Test 1: System instruction
    print("\nTest 1: System Instruction")
    print("-"*80)
    print(template.get_system_instruction())
    
    # Test 2: RAG prompt
    print("\n" + "="*80)
    print("Test 2: RAG Prompt")
    print("="*80)
    rag_prompt = template.format_rag_prompt(test_query, test_context)
    print(rag_prompt)
    
    # Test 3: Simple prompt
    print("\n" + "="*80)
    print("Test 3: Simple Prompt")
    print("="*80)
    simple_prompt = template.format_simple_prompt(test_query, test_context)
    print(simple_prompt)
    
    # Test 4: Conversational prompt
    print("\n" + "="*80)
    print("Test 4: Conversational Prompt")
    print("="*80)
    history = [
        {'role': 'user', 'content': 'What is AI?'},
        {'role': 'assistant', 'content': 'AI is artificial intelligence.'}
    ]
    conv_prompt = template.format_conversational_prompt(test_query, test_context, history)
    print(conv_prompt)
    
    # Test 5: Extraction prompt
    print("\n" + "="*80)
    print("Test 5: Extraction Prompt")
    print("="*80)
    extract_prompt = template.format_extraction_prompt(
        "What are the three types of machine learning?",
        test_context
    )
    print(extract_prompt)
    
    # Test 6: Summary prompt
    print("\n" + "="*80)
    print("Test 6: Summary Prompt")
    print("="*80)
    summary_prompt = template.format_summary_prompt(test_context, focus="types of learning")
    print(summary_prompt)
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)