"""
Chatbot tools package.
Contains all the tools available to the ReAct agent for Sprint 1 and advanced tools for Sprint 2.
"""

from .legal_search_tool import LegalSearchTool
from .summarizer_tool import BasicSummarizerTool
from .validator_tool import ValidatorTool
from .advanced import CaseComparatorTool, PrecedentExplorerTool, CitationGeneratorTool

def get_available_tools():
    """Get all available tools for the chatbot agent."""
    return [
        LegalSearchTool(),
        BasicSummarizerTool(),
        ValidatorTool(),
        CaseComparatorTool(),
        PrecedentExplorerTool(), 
        CitationGeneratorTool(),
    ]

__all__ = [
    "LegalSearchTool",
    "BasicSummarizerTool", 
    "ValidatorTool",
    "CaseComparatorTool",
    "PrecedentExplorerTool",
    "CitationGeneratorTool",
    "get_available_tools",
]