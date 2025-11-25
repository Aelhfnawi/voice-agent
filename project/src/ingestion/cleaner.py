"""
Text cleaning module for the Technical RAG Assistant.

This module handles cleaning and normalization of extracted text:
- Remove unwanted characters and artifacts
- Normalize whitespace
- Remove URLs and emails (optional)
- Fix encoding issues
- Remove page numbers and headers/footers
- Handle special technical formatting

The cleaner preserves technical content like code snippets and formulas.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans and normalizes extracted text for better chunking and embedding.
    
    Designed to preserve technical content while removing noise.
    """
    
    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = False,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = True,
        min_text_length: int = 50
    ):
        """
        Initialize the TextCleaner.
        
        Args:
            remove_urls: Whether to remove URLs from text
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
            remove_special_chars: Whether to remove special characters (preserves technical content)
            min_text_length: Minimum length of text to keep (in characters)
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.min_text_length = min_text_length
        
        logger.info("TextCleaner initialized")
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        original_length = len(text)
        
        # Apply cleaning steps
        text = self._remove_null_bytes(text)
        text = self._fix_encoding_issues(text)
        text = self._remove_page_artifacts(text)
        
        if self.remove_urls:
            text = self._remove_urls(text)
        
        if self.remove_emails:
            text = self._remove_emails(text)
        
        if self.remove_special_chars:
            text = self._remove_special_characters(text)
        
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        text = self._remove_repeated_punctuation(text)
        text = self._fix_line_breaks(text)
        text = text.strip()
        
        # Check minimum length
        if len(text) < self.min_text_length:
            logger.debug(f"Text too short after cleaning: {len(text)} chars (min: {self.min_text_length})")
            return ""
        
        cleaned_length = len(text)
        reduction = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
        
        logger.debug(f"Cleaned text: {original_length} -> {cleaned_length} chars ({reduction:.1f}% reduction)")
        
        return text
    
    def _remove_null_bytes(self, text: str) -> str:
        """Remove null bytes and other problematic characters."""
        return text.replace('\x00', '').replace('\ufffd', '')
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues."""
        # Replace common encoding artifacts
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Â': '',
            'â€¦': '...',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _remove_page_artifacts(self, text: str) -> str:
        """
        Remove page numbers, headers, and footers.
        
        Common patterns:
        - Page X of Y
        - Page X
        - Standalone numbers at line start/end
        """
        # Remove "Page X" or "Page X of Y" patterns
        text = re.sub(r'\bPage\s+\d+(?:\s+of\s+\d+)?\b', '', text, flags=re.IGNORECASE)
        
        # Remove standalone page numbers at start of lines
        text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove common header/footer separators
        text = re.sub(r'^[_\-=]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # Match http(s) URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Match www URLs
        text = re.sub(r'www\.[^\s]+', '', text)
        
        return text
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """
        Remove special characters while preserving technical content.
        
        Preserves:
        - Alphanumeric characters
        - Common punctuation (. , ! ? ; : ' " - )
        - Technical symbols (+ - * / = < > # $ % & @)
        - Parentheses and brackets
        - Newlines and tabs
        """
        # Keep: letters, numbers, punctuation, technical symbols, whitespace
        # Remove: control characters, exotic unicode, etc.
        text = re.sub(r'[^\w\s.,!?;:\'\"\-\(\)\[\]\{\}+=<>/*#$%&@\n\t]', ' ', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace.
        
        - Replace multiple spaces with single space
        - Replace multiple newlines with double newline (paragraph break)
        - Remove trailing whitespace from lines
        """
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Replace multiple spaces with single space (but preserve newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace 3+ newlines with 2 newlines (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _remove_repeated_punctuation(self, text: str) -> str:
        """Remove repeated punctuation (e.g., '...' -> '.', '!!!' -> '!')."""
        # Keep intentional ellipsis (...)
        text = re.sub(r'\.{4,}', '...', text)
        
        # Remove other repeated punctuation
        text = re.sub(r'([!?;:,])\1{2,}', r'\1', text)
        
        return text
    
    def _fix_line_breaks(self, text: str) -> str:
        """
        Fix awkward line breaks in the middle of sentences.
        
        Joins lines that were broken mid-sentence (common in PDFs).
        """
        # Join lines that end with lowercase and next starts with lowercase
        # (indicates broken sentence)
        lines = text.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                fixed_lines.append('')
                i += 1
                continue
            
            # Check if this line should be joined with next
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                
                # Join if current ends with lowercase and next starts with lowercase
                # and current doesn't end with sentence-ending punctuation
                if (current_line and next_line and
                    current_line[-1].islower() and
                    next_line[0].islower() and
                    not current_line.endswith(('.', '!', '?', ':', ';'))):
                    
                    fixed_lines.append(current_line + ' ' + next_line)
                    i += 2
                    continue
            
            fixed_lines.append(current_line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def clean_batch(self, texts: list[str]) -> list[str]:
        """
        Clean multiple texts.
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]
    
    def get_cleaning_stats(self, original: str, cleaned: str) -> dict:
        """
        Get statistics about the cleaning process.
        
        Args:
            original: Original text
            cleaned: Cleaned text
            
        Returns:
            Dictionary with cleaning statistics
        """
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'chars_removed': len(original) - len(cleaned),
            'reduction_percent': round((len(original) - len(cleaned)) / len(original) * 100, 2) if len(original) > 0 else 0,
            'original_lines': original.count('\n') + 1,
            'cleaned_lines': cleaned.count('\n') + 1,
        }


if __name__ == "__main__":
    # Test the cleaner
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test text with various issues
    test_text = """
    Page 1 of 10
    
    This is a  test   document with    multiple    spaces.
    
    
    
    This paragraph has too many line breaks.
    
    Visit our website at https://example.com for more info.
    Contact us at test@example.com
    
    This line ends with lower
    case and continues here.
    
    Some repeated punctuation!!!
    And some ellipsis.......
    
    Special characters: â€™ â€œ â€
    
    _______________
    
    Page 2 of 10
    
    More content here.
    """
    
    cleaner = TextCleaner()
    
    print("="*80)
    print("ORIGINAL TEXT:")
    print("="*80)
    print(test_text)
    print("\n" + "="*80)
    print("CLEANED TEXT:")
    print("="*80)
    
    cleaned = cleaner.clean(test_text)
    print(cleaned)
    
    print("\n" + "="*80)
    print("CLEANING STATS:")
    print("="*80)
    
    stats = cleaner.get_cleaning_stats(test_text, cleaned)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("="*80)