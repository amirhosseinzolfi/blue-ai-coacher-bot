# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Jira MCP Integration:** Full integration with Jira via Model Context Protocol (MCP)
  - Automatic detection of Jira-related queries using Persian and English keywords
  - ReAct agent with MCP tools for real-time task and sprint management
  - Support for querying tasks, sprints, boards, and issues
  - Persian language support for Jira interactions
- **Async SQLite Support:** Upgraded to AsyncSqliteSaver for async LangGraph operations
- **Enhanced Prompts:** Added Jira-specific system prompts and keywords in Persian
- **Improved Configuration:** Added JIRA_ENABLED, JIRA_MCP_URL, and JIRA_DEFAULT_PROJECT settings
- **Dependencies:** Added langchain-mcp-adapters, mcp, and aiosqlite packages

### Changed
- Updated requirements.txt with organized categories and version pinning
- Enhanced .gitignore with comprehensive file patterns
- Updated README.md with Jira integration documentation
- Improved langgraph_code.py to use AsyncSqliteSaver for async operations
- Enhanced graph_definition.py with Jira tool loading and ReAct agent support

### Fixed
- Fixed async/await compatibility issues in LangGraph execution
- Corrected SQLite checkpointer initialization for async operations

## [Previous Versions]

### Core Features
- Personalized business coaching via Telegram
- Task management and daily reporting
- Instagram content generation
- Image analysis and generation
- Business information summarization
- Team leaderboard generation
- Customizable AI tone (Friendly, Professional, Creative)
- Persian language support
- SQLite database management
- LangChain & LangGraph integration
