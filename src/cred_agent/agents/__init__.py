"""CrewAI agents for the Cred Domain Support Agent."""

from src.cred_agent.agents.retrieval_agent import (
    RETRIEVAL_SIMILARITY_THRESHOLD,
    CredKnowledgeRetrievalTool,
    create_retrieval_agent,
    load_fixed_size_retriever,
)
from src.cred_agent.agents.lookup_agent import (
    CredLoanApplicationLookupTool,
    LoanApplicationRecordId,
    create_lookup_agent,
)
from src.cred_agent.agents.response_composer import (
    create_response_composer_agent,
    create_response_composer_task,
)

__all__ = [
    "RETRIEVAL_SIMILARITY_THRESHOLD",
    "CredKnowledgeRetrievalTool",
    "create_retrieval_agent",
    "load_fixed_size_retriever",
    "CredLoanApplicationLookupTool",
    "LoanApplicationRecordId",
    "create_lookup_agent",
    "create_response_composer_agent",
    "create_response_composer_task",
]
