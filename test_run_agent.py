#!/usr/bin/env python3
"""
Test script for run_agent function with modified SqliteSaver.
"""
import os
import sys
import logging
from langgraph.checkpoint.sqlite import SqliteSaver

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Setting up test environment...")

# Initialize SqliteSaver explicitly
memory_saver = SqliteSaver(connection_string="checkpoints.sqlite")
logger.info("Initialized SqliteSaver")

# Import run_agent after setting up SqliteSaver
from langgraph_code import run_agent

# Test parameters
chat_id = "12345"
message_id = "test_message_id"
query = "This is a test message to check if the agent works correctly with the fixed SqliteSaver."

logger.info(f"Running agent with chat_id: {chat_id}")
try:
    response = run_agent(query, chat_id, message_id)
    logger.info(f"Agent response: {response}")
    print("TEST SUCCESSFUL")
except Exception as e:
    logger.error(f"Error running agent: {e}", exc_info=True)
    print("TEST FAILED")
