import re
import logging
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COMPANY_LIST = [
    "google", "apple", "microsoft", "amazon", "facebook", "tesla",
    "oracle", "ibm", "intel", "nvidia", "qualcomm", "sap",
    "salesforce", "uber", "airbnb", "twitter", "meta", "snap",
    "zoom", "palantir"
]

class TextProcessingConfig(BaseModel):
    """Configuration for text preprocessing parameters."""
    company_list: List[str] = Field(default=COMPANY_LIST, description="List of company names to mask")
    mask_numbers: bool = Field(default=True, description="Whether to mask numeric characters")
    mask_companies: bool = Field(default=True, description="Whether to mask company names")
    
    @field_validator('company_list')
    def validate_company_list(cls, v: List[str]) -> List[str]:
        if not v:
            logger.warning("Empty company list provided, company masking will have no effect")
        return [company.lower() for company in v]

def preprocess(text: str) -> str:
    """
    Preprocess text by converting to lowercase and normalizing whitespace.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    try:
        if not isinstance(text, str):
            logger.warning(f"Expected string input, got {type(text)}. Converting to string.")
            text = str(text)
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return text

def mask_text(text: str, config: Optional[TextProcessingConfig] = None) -> str:
    """
    Mask sensitive information in text.
    
    Args:
        text: Input text to mask
        config: Configuration for text masking
        
    Returns:
        Masked text
    """
    try:
        if not isinstance(text, str):
            logger.warning(f"Expected string input, got {type(text)}. Converting to string.")
            text = str(text)
            
        # Use default config if none provided
        if config is None:
            config = TextProcessingConfig()
            
        t = text.replace("\n", " ").strip().lower()
        
        # Mask numbers if configured
        if config.mask_numbers:
            t = ''.join('[NUM]' if ch.isdigit() else ch for ch in t)
        
        # Mask company names if configured
        if config.mask_companies:
            for comp in config.company_list:
                t = re.sub(rf"\b{comp}\b", '[COMPANY]', t)
                
        return t
    except Exception as e:
        logger.error(f"Error masking text: {str(e)}")
        return text
