import random, re, string
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Callable, Optional, Annotated
from pydantic import BaseModel, Field, field_validator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load stopwords once
stop_words = set(stopwords.words('english'))

class AugmentationConfig(BaseModel):
    """Configuration for text augmentation parameters."""
    synonym_rate: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.01, description="Rate of synonym replacement")
    html_entity_prob: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.1, description="Probability of HTML entity replacement")
    word_dup_prob: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.05, description="Probability of word duplication")
    case_swap_prob: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.1, description="Probability of case swapping")
    punct_space_prob: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.05, description="Probability of punctuation/space insertion")
    truncate_prob: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.1, description="Probability of text truncation")
    char_swap_prob: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.02, description="Probability of character swapping")
    min_augmentations: int = Field(default=1, ge=1, description="Minimum number of augmentations to apply")
    max_augmentations: int = Field(default=5, ge=1, description="Maximum number of augmentations to apply")
    
    @field_validator('max_augmentations')
    def validate_max_augmentations(cls, v: int, info) -> int:
        if v < info.data.get('min_augmentations', 1):
            raise ValueError("max_augmentations must be >= min_augmentations")
        return v


def get_synonyms(word: str) -> List[str]:
    """Return a list of synonyms for a word from WordNet (excluding itself)."""
    try:
        syns = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    syns.add(name)
        return list(syns)
    except Exception as e:
        logger.warning(f"Error getting synonyms for '{word}': {e}")
        return []


def synonym_replacement_rate(text: str, rate: float = 0.01) -> str:
    """
    Replace approximately `rate` fraction of non-stopwords in the text with synonyms.

    Args:
        text (str): Input text (sentence or larger).
        rate (float): Fraction of replaceable words to swap out (default 0.01 = 1%).

    Returns:
        str: Augmented text.
    """
    try:
        words = word_tokenize(text)
        candidates = [i for i, w in enumerate(words)
                    if w.isalpha() and w.lower() not in stop_words]
        
        n_replace = int(len(candidates) * rate)
        if n_replace < 1:
            return text  
        
        random.shuffle(candidates)
        replaced = 0
        for idx in candidates:
            syns = get_synonyms(words[idx])
            if syns:
                words[idx] = random.choice(syns)
                replaced += 1
            if replaced >= n_replace:
                break

        return ' '.join(words)
    except Exception as e:
        logger.warning(f"Error in synonym replacement: {e}")
        return text


def aug_html_entities(text: str, p: float = 0.1) -> str:
    """Replace characters with HTML entities with probability p."""
    try:
        entities = {"'": "&#39;", '"': "&quot;", "&": "&amp;"}
        for ch, ent in entities.items():
            if random.random() < p:
                text = text.replace(ch, ent)
        return text
    except Exception as e:
        logger.warning(f"Error in HTML entity augmentation: {e}")
        return text


def aug_word_dup(text: str, p: float = 0.05) -> str:
    """Duplicate a random word with probability p."""
    try:
        words = text.split()
        if words and random.random() < p:
            i = random.randrange(len(words))
            words.insert(i, words[i])
        return " ".join(words)
    except Exception as e:
        logger.warning(f"Error in word duplication: {e}")
        return text


def aug_case_swap(text: str, p: float = 0.1) -> str:
    """Randomly swap case of characters with probability p."""
    try:
        return "".join(c.upper() if random.random() < p else c for c in text)
    except Exception as e:
        logger.warning(f"Error in case swapping: {e}")
        return text


def aug_punct_space(text: str, p: float = 0.05) -> str:
    """Insert random punctuation after characters with probability p."""
    try:
        out = []
        for c in text:
            if c.isalnum() and random.random() < p:
                out.append(c + random.choice(string.punctuation))
            else:
                out.append(c)
        s = "".join(out)
        return re.sub(r" ", lambda m: " " + (" " if random.random() < p else ""), s)
    except Exception as e:
        logger.warning(f"Error in punctuation/space augmentation: {e}")
        return text


def aug_truncate(text: str, p: float = 0.1) -> str:
    """Truncate text with probability p."""
    try:
        if random.random() < p and len(text) > 20:
            cut = int(len(text) * random.uniform(0.7, 0.9))
            return text[:cut]
        return text
    except Exception as e:
        logger.warning(f"Error in text truncation: {e}")
        return text


def aug_char_swap(text: str, p: float = 0.02) -> str:
    """Swap adjacent characters with probability p."""
    try:
        chars = list(text)
        for i in range(len(chars) - 1):
            if random.random() < p:
                chars[i], chars[i+1] = chars[i+1], chars[i]
        return "".join(chars)
    except Exception as e:
        logger.warning(f"Error in character swapping: {e}")
        return text


def augment_text(text: str, config: Optional[AugmentationConfig] = None) -> str:
    """
    Apply multiple random augmentations to the input text.
    
    Args:
        text (str): Input text to augment
        config (AugmentationConfig, optional): Configuration for augmentation parameters
        
    Returns:
        str: Augmented text
    """
    if not text:
        return text
        
    try:
        # Use default config if none provided
        if config is None:
            config = AugmentationConfig()
            
        aug_funcs: List[Callable] = [
            lambda t: aug_html_entities(t, config.html_entity_prob),
            lambda t: aug_word_dup(t, config.word_dup_prob),
            lambda t: aug_case_swap(t, config.case_swap_prob),
            lambda t: aug_punct_space(t, config.punct_space_prob),
            lambda t: aug_truncate(t, config.truncate_prob),
            lambda t: aug_char_swap(t, config.char_swap_prob),
            lambda t: synonym_replacement_rate(t, config.synonym_rate)
        ]

        n = random.randint(config.min_augmentations, config.max_augmentations)
        chosen = random.sample(aug_funcs, k=min(n, len(aug_funcs)))

        result = text
        for fn in chosen:
            result = fn(result)
        return result
    except Exception as e:
        logger.error(f"Error during text augmentation: {e}")
        return text  # Return original text in case of error