"""
Text chunking module for the Technical RAG Assistant.

This module handles intelligent text chunking:
- Semantic chunking (respects sentence and paragraph boundaries)
- Fixed-size chunking with overlap
- Recursive chunking for large documents
- Preserves context with overlap

The chunker is optimized for technical documents and code.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into chunks for embedding and retrieval.
    
    Uses intelligent splitting that respects sentence boundaries
    and maintains context through overlap.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the TextChunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size to keep (discard smaller)
            separators: List of separators to use for splitting (in priority order)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Default separators in priority order
        if separators is None:
            self.separators = [
                "\n\n",      # Paragraph breaks (highest priority)
                "\n",        # Line breaks
                ". ",        # Sentence ends
                "! ",        # Exclamation sentences
                "? ",        # Question sentences
                "; ",        # Semicolons
                ", ",        # Commas
                " ",         # Words
                ""           # Characters (fallback)
            ]
        else:
            self.separators = separators
        
        logger.info(f"TextChunker initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = text.strip()
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            logger.debug(f"Text smaller than chunk size, returning single chunk")
            return [text]
        
        # Perform recursive chunking
        chunks = self._recursive_split(text, self.separators)
        
        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_size]
        
        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators in priority order.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of chunks
        """
        # Base case: no separators left, split by characters
        if not separators:
            return self._split_by_characters(text)
        
        # Get current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # If separator is empty, split by characters
        if separator == "":
            return self._split_by_characters(text)
        
        # Split by current separator
        splits = text.split(separator)
        
        # Rebuild splits with separator (except last)
        if separator != "":
            splits = [split + separator if i < len(splits) - 1 else split 
                     for i, split in enumerate(splits)]
        
        # Merge splits into chunks
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If split is empty, skip
            if not split.strip():
                continue
            
            # If adding this split would exceed chunk size
            if len(current_chunk) + len(split) > self.chunk_size:
                # If current chunk is not empty, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text
                
                # If split itself is too large, recursively split it
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    
                    # Add first sub-chunk to current chunk
                    if sub_chunks:
                        if current_chunk:
                            current_chunk += sub_chunks[0]
                        else:
                            current_chunk = sub_chunks[0]
                        
                        # Add remaining sub-chunks
                        if len(sub_chunks) > 1:
                            chunks.append(current_chunk.strip())
                            chunks.extend([chunk.strip() for chunk in sub_chunks[1:-1]])
                            current_chunk = sub_chunks[-1]
                else:
                    current_chunk += split
            else:
                current_chunk += split
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """
        Split text by characters (fallback method).
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """
        Get overlap text from the end of a chunk.
        
        Args:
            text: Text to get overlap from
            
        Returns:
            Overlap text
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        # Get last chunk_overlap characters
        overlap = text[-self.chunk_overlap:]
        
        # Try to start at a word boundary
        space_idx = overlap.find(' ')
        if space_idx != -1:
            overlap = overlap[space_idx:].strip()
        
        return overlap
    
    def chunk_with_metadata(self, text: str, doc_metadata: Optional[dict] = None) -> List[dict]:
        """
        Chunk text and attach metadata to each chunk.
        
        Args:
            text: Text to chunk
            doc_metadata: Metadata to attach to all chunks
            
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        chunks = self.chunk(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'metadata': {
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    **(doc_metadata or {})
                }
            }
            result.append(chunk_data)
        
        return result
    
    def get_chunk_stats(self, chunks: List[str]) -> dict:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'num_chunks': 0,
                'total_chars': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        return {
            'num_chunks': len(chunks),
            'total_chars': sum(chunk_sizes),
            'avg_chunk_size': round(sum(chunk_sizes) / len(chunks), 2),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }


class SemanticChunker(TextChunker):
    """
    Enhanced chunker that uses semantic boundaries.
    
    Specifically designed for technical documents with sections, code blocks, etc.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100):
        """Initialize SemanticChunker."""
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
    
    def chunk(self, text: str) -> List[str]:
        """
        Chunk text using semantic boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        if not text or not text.strip():
            return []
        
        # First, try to identify sections
        sections = self._identify_sections(text)
        
        if len(sections) > 1:
            # Chunk each section separately
            all_chunks = []
            for section in sections:
                section_chunks = super().chunk(section)
                all_chunks.extend(section_chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
            return all_chunks
        else:
            # No clear sections, use regular chunking
            return super().chunk(text)
    
    def _identify_sections(self, text: str) -> List[str]:
        """
        Identify sections in text (headers, major breaks).
        
        Args:
            text: Text to analyze
            
        Returns:
            List of sections
        """
        # Look for common section patterns
        patterns = [
            r'\n#{1,6}\s+.+\n',           # Markdown headers
            r'\n[A-Z][A-Za-z\s]{3,50}:\n', # Title: format
            r'\n\d+\.\s+[A-Z].+\n',       # 1. Numbered sections
            r'\n[IVX]+\.\s+[A-Z].+\n',    # I. Roman numerals
        ]
        
        # Find section boundaries
        section_starts = [0]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start = match.start()
                if start not in section_starts:
                    section_starts.append(start)
        
        section_starts.append(len(text))
        section_starts.sort()
        
        # Extract sections
        sections = []
        for i in range(len(section_starts) - 1):
            start = section_starts[i]
            end = section_starts[i + 1]
            section = text[start:end].strip()
            
            if section:
                sections.append(section)
        
        return sections if len(sections) > 1 else [text]


if __name__ == "__main__":
    # Test the chunker
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test text
    test_text = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. These systems improve their performance over time without being explicitly programmed for every scenario.
    
    The main types of machine learning include supervised learning, unsupervised learning, and reinforcement learning. Each type has its own use cases and advantages.
    
    Supervised Learning
    
    In supervised learning, the algorithm learns from labeled data. The goal is to learn a mapping from inputs to outputs based on example input-output pairs. Common applications include image classification, spam detection, and price prediction.
    
    Unsupervised Learning
    
    Unsupervised learning works with unlabeled data. The algorithm tries to find patterns and structures in the data without guidance. Clustering and dimensionality reduction are common unsupervised learning tasks.
    
    Reinforcement Learning
    
    Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward over time.
    """
    
    print("="*80)
    print("ORIGINAL TEXT:")
    print("="*80)
    print(test_text)
    print(f"\nText length: {len(test_text)} characters")
    
    # Test regular chunker
    print("\n" + "="*80)
    print("REGULAR CHUNKING (chunk_size=300, overlap=50):")
    print("="*80)
    
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk(test_text)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    stats = chunker.get_chunk_stats(chunks)
    print("\n" + "="*80)
    print("CHUNKING STATS:")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test semantic chunker
    print("\n" + "="*80)
    print("SEMANTIC CHUNKING:")
    print("="*80)
    
    semantic_chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
    semantic_chunks = semantic_chunker.chunk(test_text)
    
    print(f"\nCreated {len(semantic_chunks)} semantic chunks")
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"\n--- Semantic Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print("="*80)