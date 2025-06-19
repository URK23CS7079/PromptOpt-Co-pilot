"""
PromptOpt Co-Pilot Prompt Utilities Module

This module provides essential functions for prompt processing, validation, template handling,
and text manipulation used throughout the optimization pipeline.

Author: PromptOpt Co-Pilot Team
Version: 1.0.0
"""

import re
import string
import difflib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Callable, Any, Tuple
from functools import lru_cache
from enum import Enum
import hashlib
import json


class PromptFormat(Enum):
    """Supported prompt formats."""
    COMPLETION = "completion"
    CHAT = "chat"
    INSTRUCTION = "instruction"
    SYSTEM_USER = "system_user"


class IssueType(Enum):
    """Types of prompt issues."""
    SYNTAX_ERROR = "syntax_error"
    MISSING_VARIABLE = "missing_variable"
    UNUSED_VARIABLE = "unused_variable"
    TOKEN_LIMIT = "token_limit"
    COMPLEXITY = "complexity"
    QUALITY = "quality"
    FORMATTING = "formatting"


class SeverityLevel(Enum):
    """Severity levels for issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_suggestion(self, message: str) -> None:
        """Add a suggestion."""
        self.suggestions.append(message)


@dataclass
class Issue:
    """Detected issue in prompt."""
    type: IssueType
    severity: SeverityLevel
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class Suggestion:
    """Improvement suggestion for prompt."""
    category: str
    message: str
    rationale: str
    priority: int = 1  # 1 = high, 5 = low
    example: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float  # 0-100
    readability_score: float
    clarity_score: float
    completeness_score: float
    token_count: int
    estimated_cost: Dict[str, float]
    issues: List[Issue]
    suggestions: List[Suggestion]
    complexity_metrics: Dict[str, Any]


class PromptTemplate:
    """Template management class for prompts."""
    
    def __init__(self, template: str, name: str = ""):
        self.template = template
        self.name = name
        self.variables = self._extract_variables()
        self.schema = self._build_schema()
    
    def _extract_variables(self) -> List[str]:
        """Extract template variables from template string."""
        pattern = r'\{\{([^}]+)\}\}'
        matches = re.findall(pattern, self.template)
        return [var.strip() for var in matches]
    
    def _build_schema(self) -> Dict[str, Dict[str, Any]]:
        """Build schema for template variables."""
        schema = {}
        for var in self.variables:
            # Parse variable with optional type hints
            if ':' in var:
                var_name, var_type = var.split(':', 1)
                var_name = var_name.strip()
                var_type = var_type.strip()
            else:
                var_name = var
                var_type = "string"
            
            schema[var_name] = {
                "type": var_type,
                "required": True,
                "description": f"Variable {var_name}"
            }
        return schema
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with provided variables."""
        rendered = self.template
        for var_name, value in variables.items():
            pattern = r'\{\{\s*' + re.escape(var_name) + r'(?::[^}]*)?\s*\}\}'
            rendered = re.sub(pattern, str(value), rendered)
        return rendered
    
    def validate_variables(self, variables: Dict[str, Any]) -> ValidationResult:
        """Validate provided variables against schema."""
        result = ValidationResult(is_valid=True)
        
        # Check for missing required variables
        for var_name, var_info in self.schema.items():
            if var_info.get("required", True) and var_name not in variables:
                result.add_error(f"Missing required variable: {var_name}")
        
        # Check for unused variables
        for var_name in variables:
            if var_name not in self.schema:
                result.add_warning(f"Unused variable: {var_name}")
        
        return result


class PromptValidator:
    """Comprehensive prompt validation with configurable rules."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default validation configuration."""
        return {
            "max_tokens": 4000,
            "min_length": 10,
            "max_length": 50000,
            "allow_empty_lines": True,
            "require_instruction": True,
            "check_grammar": False,
            "forbidden_patterns": [],
            "required_patterns": []
        }
    
    def validate(self, prompt: str) -> ValidationResult:
        """Comprehensive prompt validation."""
        result = ValidationResult(is_valid=True)
        
        # Basic checks
        self._check_length(prompt, result)
        self._check_syntax(prompt, result)
        self._check_formatting(prompt, result)
        self._check_patterns(prompt, result)
        self._check_variables(prompt, result)
        
        return result
    
    def _check_length(self, prompt: str, result: ValidationResult) -> None:
        """Check prompt length constraints."""
        length = len(prompt)
        if length < self.config["min_length"]:
            result.add_error(f"Prompt too short: {length} < {self.config['min_length']}")
        elif length > self.config["max_length"]:
            result.add_error(f"Prompt too long: {length} > {self.config['max_length']}")
    
    def _check_syntax(self, prompt: str, result: ValidationResult) -> None:
        """Check for syntax errors in prompt."""
        # Check for unmatched braces
        open_braces = prompt.count('{')
        close_braces = prompt.count('}')
        if open_braces != close_braces:
            result.add_error(f"Unmatched braces: {open_braces} open, {close_braces} close")
        
        # Check for malformed variables
        pattern = r'\{[^}]*\{|\}[^{]*\}'
        if re.search(pattern, prompt):
            result.add_error("Malformed variable syntax detected")
    
    def _check_formatting(self, prompt: str, result: ValidationResult) -> None:
        """Check prompt formatting."""
        lines = prompt.split('\n')
        
        # Check for excessive empty lines
        if not self.config["allow_empty_lines"]:
            for i, line in enumerate(lines):
                if not line.strip():
                    result.add_warning(f"Empty line at line {i + 1}")
        
        # Check for trailing whitespace
        for i, line in enumerate(lines):
            if line.endswith(' ') or line.endswith('\t'):
                result.add_warning(f"Trailing whitespace at line {i + 1}")
    
    def _check_patterns(self, prompt: str, result: ValidationResult) -> None:
        """Check for forbidden and required patterns."""
        # Check forbidden patterns
        for pattern in self.config.get("forbidden_patterns", []):
            if re.search(pattern, prompt, re.IGNORECASE):
                result.add_error(f"Forbidden pattern found: {pattern}")
        
        # Check required patterns
        for pattern in self.config.get("required_patterns", []):
            if not re.search(pattern, prompt, re.IGNORECASE):
                result.add_warning(f"Required pattern missing: {pattern}")
    
    def _check_variables(self, prompt: str, result: ValidationResult) -> None:
        """Check template variables."""
        variables = extract_variables(prompt)
        
        # Check for duplicate variables
        seen = set()
        for var in variables:
            if var in seen:
                result.add_warning(f"Duplicate variable: {var}")
            seen.add(var)


# Core Prompt Processing Functions

def clean_prompt(prompt: str) -> str:
    """
    Clean and normalize prompt formatting.
    
    Args:
        prompt: Raw prompt string
        
    Returns:
        Cleaned prompt string
    """
    if not prompt:
        return ""
    
    # Remove excessive whitespace
    prompt = re.sub(r'\s+', ' ', prompt.strip())
    
    # Normalize line breaks
    prompt = re.sub(r'\n\s*\n\s*\n+', '\n\n', prompt)
    
    # Remove trailing whitespace from lines
    lines = prompt.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    
    return '\n'.join(cleaned_lines)


def extract_variables(prompt: str) -> List[str]:
    """
    Extract template variables from prompt.
    
    Args:
        prompt: Prompt string with variables in {{variable}} format
        
    Returns:
        List of variable names
    """
    pattern = r'\{\{([^}]+)\}\}'
    matches = re.findall(pattern, prompt)
    
    # Clean and deduplicate variables
    variables = []
    seen = set()
    for match in matches:
        # Handle variables with type hints (e.g., {{name:string}})
        var_name = match.split(':')[0].strip()
        if var_name not in seen:
            variables.append(var_name)
            seen.add(var_name)
    
    return variables


def validate_prompt_syntax(prompt: str) -> ValidationResult:
    """
    Validate prompt syntax.
    
    Args:
        prompt: Prompt string to validate
        
    Returns:
        ValidationResult with errors and warnings
    """
    validator = PromptValidator()
    return validator.validate(prompt)


def normalize_prompt(prompt: str) -> str:
    """
    Standardize prompt format.
    
    Args:
        prompt: Raw prompt string
        
    Returns:
        Normalized prompt string
    """
    if not prompt:
        return ""
    
    # Clean the prompt first
    prompt = clean_prompt(prompt)
    
    # Ensure proper spacing around variables
    prompt = re.sub(r'\{\{([^}]+)\}\}', r'{{ \1 }}', prompt)
    
    # Standardize punctuation spacing
    prompt = re.sub(r'\s*([.!?])\s*', r'\1 ', prompt)
    
    # Ensure single space after colons
    prompt = re.sub(r':\s*', ': ', prompt)
    
    return prompt.strip()


def truncate_prompt(prompt: str, max_tokens: int, tokenizer: Callable[[str], List[str]]) -> str:
    """
    Truncate prompt to fit within token limit.
    
    Args:
        prompt: Prompt string to truncate
        max_tokens: Maximum number of tokens allowed
        tokenizer: Function to tokenize text
        
    Returns:
        Truncated prompt string
    """
    if not prompt:
        return ""
    
    tokens = tokenizer(prompt)
    if len(tokens) <= max_tokens:
        return prompt
    
    # Truncate tokens and rejoin
    truncated_tokens = tokens[:max_tokens]
    
    # Try to truncate at sentence boundaries
    truncated_text = " ".join(truncated_tokens)
    
    # Find the last complete sentence
    sentences = re.split(r'[.!?]+', truncated_text)
    if len(sentences) > 1:
        # Remove the last incomplete sentence
        complete_sentences = sentences[:-1]
        return ".".join(complete_sentences) + "."
    
    return truncated_text


def merge_prompts(prompts: List[str], strategy: str = "concatenate") -> str:
    """
    Combine multiple prompts using specified strategy.
    
    Args:
        prompts: List of prompt strings
        strategy: Merge strategy ("concatenate", "interleave", "template")
        
    Returns:
        Merged prompt string
    """
    if not prompts:
        return ""
    
    # Clean all prompts first
    cleaned_prompts = [clean_prompt(p) for p in prompts if p.strip()]
    
    if strategy == "concatenate":
        return "\n\n".join(cleaned_prompts)
    
    elif strategy == "interleave":
        # Interleave sentences from different prompts
        all_sentences = []
        prompt_sentences = [re.split(r'[.!?]+', p) for p in cleaned_prompts]
        
        max_len = max(len(sentences) for sentences in prompt_sentences)
        for i in range(max_len):
            for sentences in prompt_sentences:
                if i < len(sentences) and sentences[i].strip():
                    all_sentences.append(sentences[i].strip())
        
        return ". ".join(all_sentences) + "."
    
    elif strategy == "template":
        # Create a template that incorporates all prompts
        template_parts = []
        for i, prompt in enumerate(cleaned_prompts):
            template_parts.append(f"## Section {i + 1}\n{prompt}")
        
        return "\n\n".join(template_parts)
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def render_template(template: str, variables: Dict[str, Any]) -> str:
    """
    Render template with variable substitution.
    
    Args:
        template: Template string with {{variable}} placeholders
        variables: Dictionary of variable values
        
    Returns:
        Rendered template string
    """
    prompt_template = PromptTemplate(template)
    return prompt_template.render(variables)


def validate_template_variables(template: str, provided_vars: Dict[str, Any]) -> ValidationResult:
    """
    Validate template variables.
    
    Args:
        template: Template string
        provided_vars: Dictionary of provided variables
        
    Returns:
        ValidationResult
    """
    prompt_template = PromptTemplate(template)
    return prompt_template.validate_variables(provided_vars)


def extract_template_schema(template: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract schema from template.
    
    Args:
        template: Template string
        
    Returns:
        Schema dictionary
    """
    prompt_template = PromptTemplate(template)
    return prompt_template.schema


def create_template_from_examples(examples: List[Dict[str, Any]]) -> str:
    """
    Infer template from example data.
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        Inferred template string
    """
    if not examples:
        return ""
    
    # Find common keys across all examples
    common_keys = set(examples[0].keys())
    for example in examples[1:]:
        common_keys &= set(example.keys())
    
    # Create template with placeholders
    template_parts = []
    for key in sorted(common_keys):
        template_parts.append(f"{{{{ {key} }}}}")
    
    return " ".join(template_parts)


@lru_cache(maxsize=1000)
def calculate_prompt_similarity(prompt1: str, prompt2: str) -> float:
    """
    Calculate semantic similarity between two prompts.
    
    Args:
        prompt1: First prompt string
        prompt2: Second prompt string
        
    Returns:
        Similarity score between 0 and 1
    """
    if not prompt1 or not prompt2:
        return 0.0
    
    # Use multiple similarity metrics and combine them
    
    # 1. Character-level similarity
    char_sim = difflib.SequenceMatcher(None, prompt1, prompt2).ratio()
    
    # 2. Word-level similarity
    words1 = set(prompt1.lower().split())
    words2 = set(prompt2.lower().split())
    
    if not words1 and not words2:
        word_sim = 1.0
    elif not words1 or not words2:
        word_sim = 0.0
    else:
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        word_sim = intersection / union
    
    # 3. Sentence structure similarity
    sentences1 = re.split(r'[.!?]+', prompt1)
    sentences2 = re.split(r'[.!?]+', prompt2)
    
    struct_sim = 1.0 - abs(len(sentences1) - len(sentences2)) / max(len(sentences1), len(sentences2), 1)
    
    # Combined similarity (weighted average)
    combined_sim = (char_sim * 0.3 + word_sim * 0.5 + struct_sim * 0.2)
    
    return round(combined_sim, 3)


def analyze_prompt_complexity(prompt: str) -> Dict[str, Any]:
    """
    Analyze prompt complexity metrics.
    
    Args:
        prompt: Prompt string to analyze
        
    Returns:
        Dictionary of complexity metrics
    """
    if not prompt:
        return {"error": "Empty prompt"}
    
    # Basic metrics
    char_count = len(prompt)
    word_count = len(prompt.split())
    sentence_count = len(re.split(r'[.!?]+', prompt))
    paragraph_count = len([p for p in prompt.split('\n\n') if p.strip()])
    
    # Vocabulary complexity
    words = prompt.lower().split()
    unique_words = len(set(words))
    vocab_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Structural complexity
    variables = extract_variables(prompt)
    variable_count = len(variables)
    
    # Readability estimate (simplified)
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    avg_chars_per_word = char_count / word_count if word_count > 0 else 0
    
    # Overall complexity score (0-100)
    complexity_score = min(100, (
        (word_count / 10) * 0.3 +
        (sentence_count / 5) * 0.2 +
        (variable_count * 5) * 0.2 +
        (vocab_diversity * 50) * 0.3
    ))
    
    return {
        "character_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "unique_words": unique_words,
        "vocabulary_diversity": round(vocab_diversity, 3),
        "variable_count": variable_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "avg_chars_per_word": round(avg_chars_per_word, 2),
        "complexity_score": round(complexity_score, 1)
    }


def detect_prompt_patterns(prompt: str) -> List[str]:
    """
    Identify common patterns in prompt.
    
    Args:
        prompt: Prompt string to analyze
        
    Returns:
        List of detected patterns
    """
    patterns = []
    
    # Common prompt patterns
    pattern_checks = [
        (r'\bplease\b', "polite_request"),
        (r'\bstep by step\b', "step_by_step"),
        (r'\bfor example\b', "examples_requested"),
        (r'\bexplain\b', "explanation_request"),
        (r'\blist\b', "list_request"),
        (r'\bcompare\b', "comparison_request"),
        (r'\bsummarize\b', "summarization_request"),
        (r'\banalyze\b', "analysis_request"),
        (r'\{([^}]+)\}', "template_variables"),
        (r'\b(you are|act as|role)\b', "role_assignment"),
        (r'\b(context|background):', "context_provided"),
        (r'\b(output|format|structure):', "format_specification"),
        (r'\b(do not|don\'t|avoid)\b', "negative_instruction"),
        (r'\b(must|should|required)\b', "requirement_specification"),
    ]
    
    for pattern, name in pattern_checks:
        if re.search(pattern, prompt, re.IGNORECASE):
            patterns.append(name)
    
    # Check for few-shot examples
    if len(re.findall(r'\bexample\s*\d+', prompt, re.IGNORECASE)) > 1:
        patterns.append("few_shot_examples")
    
    # Check for chain of thought
    if re.search(r'\bthink\b.*\bstep\b', prompt, re.IGNORECASE):
        patterns.append("chain_of_thought")
    
    return patterns


def count_tokens(prompt: str, tokenizer: Callable[[str], List[str]]) -> int:
    """
    Count tokens in prompt using provided tokenizer.
    
    Args:
        prompt: Prompt string
        tokenizer: Tokenizer function
        
    Returns:
        Number of tokens
    """
    if not prompt:
        return 0
    
    tokens = tokenizer(prompt)
    return len(tokens)


def estimate_inference_cost(prompt: str, model_config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate inference cost for prompt.
    
    Args:
        prompt: Prompt string
        model_config: Model configuration with pricing info
        
    Returns:
        Cost estimation dictionary
    """
    # Simple tokenizer for estimation (split by whitespace)
    def simple_tokenizer(text: str) -> List[str]:
        return text.split()
    
    token_count = count_tokens(prompt, simple_tokenizer)
    
    # Default pricing (per 1K tokens)
    input_cost_per_1k = model_config.get("input_cost_per_1k", 0.001)
    output_cost_per_1k = model_config.get("output_cost_per_1k", 0.002)
    
    # Estimate response length (typically 10-20% of input)
    estimated_output_tokens = token_count * 0.15
    
    input_cost = (token_count / 1000) * input_cost_per_1k
    output_cost = (estimated_output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": token_count,
        "estimated_output_tokens": int(estimated_output_tokens),
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": model_config.get("currency", "USD")
    }


def add_context(prompt: str, context: str, position: str = "before") -> str:
    """
    Add context to prompt at specified position.
    
    Args:
        prompt: Original prompt
        context: Context to add
        position: Where to add context ("before", "after", "replace")
        
    Returns:
        Modified prompt with context
    """
    if not context:
        return prompt
    
    context = clean_prompt(context)
    prompt = clean_prompt(prompt)
    
    if position == "before":
        return f"{context}\n\n{prompt}"
    elif position == "after":
        return f"{prompt}\n\n{context}"
    elif position == "replace":
        return context
    else:
        raise ValueError(f"Invalid position: {position}")


def add_examples(prompt: str, examples: List[Dict[str, Any]], format_style: str = "numbered") -> str:
    """
    Add few-shot examples to prompt.
    
    Args:
        prompt: Original prompt
        examples: List of example dictionaries
        format_style: Example format ("numbered", "bullet", "inline")
        
    Returns:
        Prompt with examples added
    """
    if not examples:
        return prompt
    
    example_text = ""
    
    if format_style == "numbered":
        for i, example in enumerate(examples, 1):
            example_text += f"\nExample {i}:\n"
            for key, value in example.items():
                example_text += f"{key}: {value}\n"
    
    elif format_style == "bullet":
        for example in examples:
            example_text += "\n• "
            example_parts = [f"{k}: {v}" for k, v in example.items()]
            example_text += " | ".join(example_parts)
    
    elif format_style == "inline":
        example_parts = []
        for example in examples:
            parts = [f"{k}: {v}" for k, v in example.items()]
            example_parts.append(f"({', '.join(parts)})")
        example_text = "\nExamples: " + ", ".join(example_parts)
    
    return f"{prompt}\n{example_text}"


def convert_prompt_style(prompt: str, target_style: str) -> str:
    """
    Convert prompt to different style.
    
    Args:
        prompt: Original prompt
        target_style: Target style ("formal", "casual", "instruction", "question")
        
    Returns:
        Converted prompt
    """
    if target_style == "formal":
        # Make more formal
        prompt = re.sub(r'\bplease\b', 'kindly', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bcan you\b', 'would you', prompt, flags=re.IGNORECASE)
        
    elif target_style == "casual":
        # Make more casual
        prompt = re.sub(r'\bkindly\b', 'please', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'\bwould you\b', 'can you', prompt, flags=re.IGNORECASE)
        
    elif target_style == "instruction":
        # Convert to imperative form
        if not prompt.strip().endswith('.'):
            prompt += '.'
        prompt = re.sub(r'^\s*please\s+', '', prompt, flags=re.IGNORECASE)
        
    elif target_style == "question":
        # Convert to question form
        if not prompt.strip().endswith('?'):
            prompt = f"Can you {prompt.strip().lower()}?"
    
    return prompt


def optimize_prompt_structure(prompt: str) -> str:
    """
    Optimize prompt structure for better performance.
    
    Args:
        prompt: Original prompt
        
    Returns:
        Optimized prompt
    """
    # Clean and normalize
    prompt = normalize_prompt(prompt)
    
    # Split into logical sections
    sections = []
    current_section = []
    
    for line in prompt.split('\n'):
        line = line.strip()
        if not line:
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
        else:
            current_section.append(line)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Reorder sections for better structure
    # 1. Context/Background
    # 2. Task description
    # 3. Examples
    # 4. Output format
    # 5. Constraints
    
    context_sections = []
    task_sections = []
    example_sections = []
    format_sections = []
    constraint_sections = []
    other_sections = []
    
    for section in sections:
        lower_section = section.lower()
        if any(word in lower_section for word in ['context', 'background', 'about']):
            context_sections.append(section)
        elif any(word in lower_section for word in ['example', 'instance', 'sample']):
            example_sections.append(section)
        elif any(word in lower_section for word in ['format', 'output', 'structure']):
            format_sections.append(section)
        elif any(word in lower_section for word in ['constraint', 'rule', 'requirement', 'must', 'should']):
            constraint_sections.append(section)
        elif any(word in lower_section for word in ['task', 'goal', 'objective', 'do', 'create', 'generate']):
            task_sections.append(section)
        else:
            other_sections.append(section)
    
    # Reconstruct optimized prompt
    optimized_sections = []
    optimized_sections.extend(context_sections)
    optimized_sections.extend(task_sections)
    optimized_sections.extend(other_sections)
    optimized_sections.extend(example_sections)
    optimized_sections.extend(format_sections)
    optimized_sections.extend(constraint_sections)
    
    return '\n\n'.join(optimized_sections)


def check_prompt_quality(prompt: str) -> QualityReport:
    """
    Comprehensive quality assessment of prompt.
    
    Args:
        prompt: Prompt string to assess
        
    Returns:
        QualityReport with detailed metrics
    """
    # Initialize components
    issues = []
    suggestions = []
    
    # Basic validation
    validation_result = validate_prompt_syntax(prompt)
    for error in validation_result.errors:
        issues.append(Issue(
            type=IssueType.SYNTAX_ERROR,
            severity=SeverityLevel.HIGH,
            message=error
        ))
    
    # Complexity analysis
    complexity = analyze_prompt_complexity(prompt)
    
    # Quality scoring
    readability_score = min(100, max(0, 100 - complexity["complexity_score"]))
    
    clarity_score = 100
    if complexity["avg_words_per_sentence"] > 20:
        clarity_score -= 20
        suggestions.append(Suggestion(
            category="clarity",
            message="Consider breaking long sentences into shorter ones",
            rationale="Shorter sentences improve readability and comprehension",
            priority=2
        ))
    
    completeness_score = 80  # Base score
    patterns = detect_prompt_patterns(prompt)
    if "examples_requested" in patterns:
        completeness_score += 10
    if "format_specification" in patterns:
        completeness_score += 10
    if "context_provided" in patterns:
        completeness_score += 5
    
    # Token counting (using simple tokenizer)
    def simple_tokenizer(text: str) -> List[str]:
        return text.split()
    
    token_count = count_tokens(prompt, simple_tokenizer)
    
    # Cost estimation
    default_model_config = {
        "input_cost_per_1k": 0.001,
        "output_cost_per_1k": 0.002,
        "currency": "USD"
    }
    estimated_cost = estimate_inference_cost(prompt, default_model_config)
    
    # Overall score calculation
    overall_score = (readability_score * 0.3 + clarity_score * 0.3 + completeness_score * 0.4)
    
    # Additional suggestions based on analysis
    if token_count > 2000:
        suggestions.append(Suggestion(
            category="efficiency",
            message="Consider reducing prompt length to improve response time",
            rationale="Shorter prompts typically result in faster processing",
            priority=3
        ))
    
    if complexity["vocabulary_diversity"] < 0.3:
        suggestions.append(Suggestion(
            category="diversity",
            message="Consider using more varied vocabulary",
            rationale="Diverse vocabulary can improve model understanding",
            priority=4
        ))
    
    if "template_variables" not in patterns and "{{" in prompt:
        issues.append(Issue(
            type=IssueType.SYNTAX_ERROR,
            severity=SeverityLevel.MEDIUM,
            message="Possible malformed template variables detected"
        ))
    
    return QualityReport(
        overall_score=round(overall_score, 1),
        readability_score=round(readability_score, 1),
        clarity_score=round(clarity_score, 1),
        completeness_score=round(completeness_score, 1),
        token_count=token_count,
        estimated_cost=estimated_cost,
        issues=issues,
        suggestions=suggestions,
        complexity_metrics=complexity
    )


def detect_potential_issues(prompt: str) -> List[Issue]:
    """
    Detect potential issues in prompt.
    
    Args:
        prompt: Prompt string to analyze
        
    Returns:
        List of detected issues
    """
    issues = []
    
    # Check for common issues
    if len(prompt.strip()) == 0:
        issues.append(Issue(
            type=IssueType.QUALITY,
            severity=SeverityLevel.CRITICAL,
            message="Empty prompt",
            suggestion="Add meaningful content to the prompt"
        ))
        return issues
    
    # Check for extremely long prompts
    if len(prompt) > 10000:
        issues.append(Issue(
            type=IssueType.TOKEN_LIMIT,
            severity=SeverityLevel.HIGH,
            message="Prompt may exceed token limits",
            suggestion="Consider breaking into smaller sections"
        ))
    
    # Check for excessive repetition
    words = prompt.lower().split()
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Only check meaningful words
            word_counts[word] = word_counts.get(word, 0) + 1
    
    for word, count in word_counts.items():
        if count > len(words) * 0.1:  # More than 10% of total words
            issues.append(Issue(
                type=IssueType.QUALITY,
                severity=SeverityLevel.MEDIUM,
                message=f"Excessive repetition of word: '{word}'",
                suggestion=f"Consider using synonyms or reducing repetition"
            ))
    
    # Check for unclear instructions
    if not any(pattern in prompt.lower() for pattern in [
        'please', 'create', 'generate', 'write', 'explain', 'describe', 'analyze', 'summarize'
    ]):
        issues.append(Issue(
            type=IssueType.QUALITY,
            severity=SeverityLevel.MEDIUM,
            message="No clear action verb found",
            suggestion="Add clear instructions about what you want the AI to do"
        ))
    
    # Check for formatting issues
    if prompt.count('(') != prompt.count(')'):
        issues.append(Issue(
            type=IssueType.FORMATTING,
            severity=SeverityLevel.LOW,
            message="Unmatched parentheses",
            suggestion="Check parentheses pairing"
        ))
    
    if prompt.count('[') != prompt.count(']'):
        issues.append(Issue(
            type=IssueType.FORMATTING,
            severity=SeverityLevel.LOW,
            message="Unmatched brackets",
            suggestion="Check bracket pairing"
        ))
    
    # Check for template variable issues
    variables = extract_variables(prompt)
    if variables:
        # Check for variables that might be undefined
        for var in variables:
            if not re.search(rf'\b{re.escape(var)}\b', prompt.replace('{{', '').replace('}}', '')):
                issues.append(Issue(
                    type=IssueType.MISSING_VARIABLE,
                    severity=SeverityLevel.MEDIUM,
                    message=f"Variable '{var}' is not explained in prompt",
                    suggestion=f"Consider explaining what '{var}' represents"
                ))
    
    return issues


def suggest_improvements(prompt: str) -> List[Suggestion]:
    """
    Generate improvement suggestions for prompt.
    
    Args:
        prompt: Prompt string to analyze
        
    Returns:
        List of improvement suggestions
    """
    suggestions = []
    
    # Analyze current prompt
    patterns = detect_prompt_patterns(prompt)
    complexity = analyze_prompt_complexity(prompt)
    
    # Structure suggestions
    if "context_provided" not in patterns:
        suggestions.append(Suggestion(
            category="structure",
            message="Consider adding context or background information",
            rationale="Context helps the AI understand the scenario better",
            priority=2,
            example="Context: You are helping a marketing team create content..."
        ))
    
    if "format_specification" not in patterns:
        suggestions.append(Suggestion(
            category="structure",
            message="Specify the desired output format",
            rationale="Clear format specifications improve output consistency",
            priority=1,
            example="Format: Provide your response as a bulleted list..."
        ))
    
    if "examples_requested" not in patterns and complexity["word_count"] > 50:
        suggestions.append(Suggestion(
            category="clarity",
            message="Consider adding examples to illustrate your request",
            rationale="Examples help clarify expectations and improve output quality",
            priority=2,
            example="For example: [provide a concrete example here]"
        ))
    
    # Complexity suggestions
    if complexity["avg_words_per_sentence"] > 25:
        suggestions.append(Suggestion(
            category="readability",
            message="Break long sentences into shorter ones",
            rationale="Shorter sentences are easier to parse and understand",
            priority=2
        ))
    
    if complexity["vocabulary_diversity"] < 0.4:
        suggestions.append(Suggestion(
            category="vocabulary",
            message="Use more varied vocabulary",
            rationale="Diverse vocabulary can improve model comprehension",
            priority=3
        ))
    
    # Specificity suggestions
    vague_words = ["good", "bad", "nice", "fine", "okay", "some", "many", "few"]
    prompt_lower = prompt.lower()
    for word in vague_words:
        if f" {word} " in prompt_lower:
            suggestions.append(Suggestion(
                category="specificity",
                message=f"Replace vague word '{word}' with more specific terms",
                rationale="Specific language leads to more precise outputs",
                priority=2
            ))
            break  # Only suggest once
    
    # Role-playing suggestions
    if "role_assignment" not in patterns and complexity["word_count"] > 100:
        suggestions.append(Suggestion(
            category="technique",
            message="Consider assigning a specific role to the AI",
            rationale="Role assignment can improve response quality and consistency",
            priority=3,
            example="You are an expert marketing strategist..."
        ))
    
    # Chain of thought suggestions
    if ("analyze" in prompt_lower or "explain" in prompt_lower) and "chain_of_thought" not in patterns:
        suggestions.append(Suggestion(
            category="technique",
            message="Ask the AI to think step by step",
            rationale="Step-by-step reasoning often produces better results",
            priority=2,
            example="Think through this step by step..."
        ))
    
    return suggestions


# Utility functions for caching and performance

@lru_cache(maxsize=500)
def _cached_clean_prompt(prompt_hash: str, prompt: str) -> str:
    """Cached version of clean_prompt for performance."""
    return clean_prompt(prompt)


def get_prompt_hash(prompt: str) -> str:
    """Generate hash for prompt caching."""
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()


def batch_process_prompts(
    prompts: List[str], 
    operation: Callable[[str], Any],
    batch_size: int = 10
) -> List[Any]:
    """
    Process prompts in batches for better performance.
    
    Args:
        prompts: List of prompt strings
        operation: Function to apply to each prompt
        batch_size: Number of prompts to process at once
        
    Returns:
        List of operation results
    """
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_results = [operation(prompt) for prompt in batch]
        results.extend(batch_results)
    
    return results


# Configuration and constants

DEFAULT_VALIDATION_CONFIG = {
    "max_tokens": 4000,
    "min_length": 10,
    "max_length": 50000,
    "allow_empty_lines": True,
    "require_instruction": True,
    "check_grammar": False,
    "forbidden_patterns": [
        r'\b(hack|exploit|bypass)\b',  # Security-related
        r'\b(illegal|harmful|dangerous)\b'  # Safety-related
    ],
    "required_patterns": []
}

PROMPT_TEMPLATES = {
    "basic_instruction": "{{instruction}}",
    "context_instruction": "Context: {{context}}\n\nInstruction: {{instruction}}",
    "few_shot": "{{context}}\n\nExamples:\n{{examples}}\n\nNow: {{instruction}}",
    "role_based": "You are {{role}}. {{context}}\n\nTask: {{instruction}}",
    "step_by_step": "{{context}}\n\nPlease complete this task step by step:\n{{instruction}}"
}

QUALITY_THRESHOLDS = {
    "excellent": 90,
    "good": 75,
    "acceptable": 60,
    "needs_improvement": 40,
    "poor": 0
}


# Export all public functions and classes
__all__ = [
    # Data classes
    'ValidationResult', 'Issue', 'Suggestion', 'QualityReport', 'PromptTemplate', 'PromptValidator',
    
    # Enums
    'PromptFormat', 'IssueType', 'SeverityLevel',
    
    # Core functions
    'clean_prompt', 'extract_variables', 'validate_prompt_syntax', 'normalize_prompt',
    'truncate_prompt', 'merge_prompts', 'render_template', 'validate_template_variables',
    'extract_template_schema', 'create_template_from_examples', 'calculate_prompt_similarity',
    'analyze_prompt_complexity', 'detect_prompt_patterns', 'count_tokens', 'estimate_inference_cost',
    'add_context', 'add_examples', 'convert_prompt_style', 'optimize_prompt_structure',
    'check_prompt_quality', 'detect_potential_issues', 'suggest_improvements',
    
    # Utility functions
    'get_prompt_hash', 'batch_process_prompts',
    
    # Constants
    'DEFAULT_VALIDATION_CONFIG', 'PROMPT_TEMPLATES', 'QUALITY_THRESHOLDS'
]


if __name__ == "__main__":
    # Example usage and testing
    sample_prompt = """
    You are a helpful assistant. Please analyze the following text and provide insights:
    
    {{text_to_analyze}}
    
    Focus on:
    1. Main themes
    2. Sentiment analysis
    3. Key insights
    
    Format your response as a structured report.
    """
    
    print("=== Prompt Utils Demo ===")
    
    # Clean and normalize
    cleaned = clean_prompt(sample_prompt)
    print(f"Cleaned prompt length: {len(cleaned)}")
    
    # Extract variables
    variables = extract_variables(sample_prompt)
    print(f"Variables found: {variables}")
    
    # Validate syntax
    validation = validate_prompt_syntax(sample_prompt)
    print(f"Validation passed: {validation.is_valid}")
    
    # Analyze complexity
    complexity = analyze_prompt_complexity(sample_prompt)
    print(f"Complexity score: {complexity['complexity_score']}")
    
    # Detect patterns
    patterns = detect_prompt_patterns(sample_prompt)
    print(f"Patterns detected: {patterns}")
    
    # Quality assessment
    quality = check_prompt_quality(sample_prompt)
    print(f"Overall quality score: {quality.overall_score}")
    print(f"Number of suggestions: {len(quality.suggestions)}")
    
    print("\n=== Demo Complete ===")