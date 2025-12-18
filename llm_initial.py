#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_initial.py - Centralized LLM Model Initialization
Handles all LLM model configurations and instance creation for the Blue AI Coacher Bot.
"""

import logging
import atexit
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import (
    GOOGLE_API_KEY,
    GOOGLE_API_KEY_2,
    OPENAI_API_KEY,
    PRIMARY_LLM_BASE_URL,
    PRIMARY_LLM_MODEL,
    PRIMARY_LLM_API_KEY,
    BUSINESS_LLM_MODEL,
    BUSINESS_LLM_API_KEY,
    USER_REPORT_LLM_MODEL,
    USER_REPORT_LLM_API_KEY,
    SUMMARY_LLM_MODEL,
    SUMMARY_LLM_API_KEY,
    IMAGE_ANALYZE_LLM_MODEL,
    IMAGE_ANALYZE_LLM_API_KEY,
    JIRA_AGENT_LLM_MODEL,
    JIRA_AGENT_LLM_API_KEY,
)

############################################
# Logger Setup
############################################
logger = logging.getLogger(__name__)

############################################
# Model Configuration (values come from config.py)
############################################
DEFAULT_TEMPERATURE = 0.5

############################################
# LLM Instance Creation Functions
############################################
def create_primary_llm(
    model: str = PRIMARY_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None
) -> ChatOpenAI:
    """Create primary LLM instance for main chat interactions."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key= PRIMARY_LLM_API_KEY or GOOGLE_API_KEY
    )

def create_business_llm(
    model: str = BUSINESS_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """Create LLM instance for business info summarization."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=api_key or BUSINESS_LLM_API_KEY or GOOGLE_API_KEY
    )

def create_user_report_llm(
    model: str = USER_REPORT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """Create LLM instance for user report generation."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=api_key or USER_REPORT_LLM_API_KEY or GOOGLE_API_KEY
    )

def create_summary_llm(
    model: str = SUMMARY_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None
) -> ChatOpenAI:
    """Create LLM instance for conversation summarization."""
    # Fallback to gpt-4o-mini if gpt-5-mini is requested (causes auth errors)
    return ChatOpenAI(
        base_url="http://localhost:15501/v1",
        model_name=model,
        temperature=temperature,
        api_key= "34534"
    )

def create_image_analyze_llm(
    model: str = IMAGE_ANALYZE_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """Create LLM instance for image analysis."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=api_key or IMAGE_ANALYZE_LLM_API_KEY or GOOGLE_API_KEY
    )

def create_jira_agent_llm(
    model: str = JIRA_AGENT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: Optional[str] = None
) -> ChatGoogleGenerativeAI:
    """Create LLM instance for Jira agent interactions."""
    return ChatOpenAI(
        base_url="http://localhost:15501/v1",
        model_name=model,
        temperature=temperature,
        api_key=api_key or JIRA_AGENT_LLM_API_KEY or OPENAI_API_KEY or "not-set"
    )

############################################
# Global LLM Instances
############################################
llm = None
llm_business = None
user_llm = None
llm_summary = None
image_analyze_llm = None
jira_agent_llm = None

def initialize_llms():
    """Initialize all LLM instances."""
    global llm, llm_business, user_llm, llm_summary, image_analyze_llm, jira_agent_llm
    
    llm = create_primary_llm()
    logger.info(f"Primary LLM initialized: {PRIMARY_LLM_MODEL}")
    
    llm_business = create_business_llm()
    logger.info(f"Business LLM initialized: {BUSINESS_LLM_MODEL}")
    
    user_llm = create_user_report_llm()
    logger.info(f"User Report LLM initialized: {USER_REPORT_LLM_MODEL}")
    
    llm_summary = create_summary_llm()
    logger.info(f"Summary LLM initialized: {SUMMARY_LLM_MODEL}")
    
    image_analyze_llm = create_image_analyze_llm()
    logger.info(f"Image Analysis LLM initialized: {IMAGE_ANALYZE_LLM_MODEL}")
    
    jira_agent_llm = create_jira_agent_llm()
    logger.info(f"Jira Agent LLM initialized: {JIRA_AGENT_LLM_MODEL}")

def cleanup_llms():
    """Cleanup LLM client connections."""
    global llm, llm_business, user_llm, llm_summary, image_analyze_llm, jira_agent_llm
    
    for llm_instance in [llm, llm_business, user_llm, llm_summary, image_analyze_llm, jira_agent_llm]:
        if llm_instance and hasattr(llm_instance, "client"):
            try:
                if callable(getattr(llm_instance.client, "close", None)):
                    llm_instance.client.close()
            except Exception as e:
                logger.error(f"Error closing LLM client: {e}")

# Register cleanup on exit
atexit.register(cleanup_llms)

# Auto-initialize on import
initialize_llms()

############################################
# Exports
############################################
__all__ = [
    "PRIMARY_LLM_MODEL",
    "BUSINESS_LLM_MODEL",
    "SUMMARY_LLM_MODEL",
    "USER_REPORT_LLM_MODEL",
    "IMAGE_ANALYZE_LLM_MODEL",
    "JIRA_AGENT_LLM_MODEL",
    "llm",
    "llm_business",
    "user_llm",
    "llm_summary",
    "image_analyze_llm",
    "jira_agent_llm",
    "create_primary_llm",
    "create_business_llm",
    "create_user_report_llm",
    "create_summary_llm",
    "create_image_analyze_llm",
    "create_jira_agent_llm",
    "initialize_llms",
    "cleanup_llms",
]
