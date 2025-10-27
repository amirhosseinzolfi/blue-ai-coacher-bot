#!/usr/bin/env python3
"""
Jira MCP Integration for Blue AI Coacher Bot
Provides Jira task management capabilities through MCP tools
"""

import asyncio
import logging
import re
from typing import List, Optional, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import JIRA_MCP_URL, JIRA_ENABLED
from utils.rich_logger import setup_logger

logger = setup_logger(level=logging.INFO, logger_name="jira_integration")

class JiraIntegration:
    def __init__(self):
        self.client = None
        self.tools = []
        self.enabled = JIRA_ENABLED
        
    async def initialize(self) -> bool:
        """Initialize Jira MCP client and load tools"""
        if not self.enabled:
            logger.info("Jira integration disabled")
            return False
            
        try:
            self.client = MultiServerMCPClient({
                "jira": {"url": JIRA_MCP_URL, "transport": "streamable_http"}
            })
            self.tools = await self.client.get_tools()
            logger.info(f"Jira MCP tools loaded: {[t.name for t in self.tools]}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Jira MCP: {e}")
            self.enabled = False
            return False
    
    def get_tools(self) -> List[Any]:
        """Get available Jira tools"""
        return self.tools if self.enabled else []
    
    def is_jira_query(self, query: str) -> bool:
        """Enhanced Jira query detection"""
        jira_keywords = [
            'jira', 'task', 'issue', 'sprint', 'board', 'epic',
            'تسک', 'کار', 'پروژه', 'اسپرینت', 'مسئله',
            'assign', 'due', 'priority', 'status', 'backlog',
            'تخصیص', 'اولویت', 'وضعیت', 'مهلت'
        ]
        jira_key_pattern = r'\b[A-Z]+-\d+\b'  # e.g., BAP-123
    
        return (
            any(keyword in query.lower() for keyword in jira_keywords) or
            bool(re.search(jira_key_pattern, query))
        )

# Global instance
jira_integration = JiraIntegration()

async def get_jira_tools() -> List[Any]:
    """Get Jira tools, initializing if needed"""
    if not jira_integration.enabled:
        return []
        
    if not jira_integration.tools:
        await jira_integration.initialize()
    
    return jira_integration.get_tools()

def is_jira_related(query: str) -> bool:
    """Check if query is Jira-related"""
    return jira_integration.is_jira_query(query)