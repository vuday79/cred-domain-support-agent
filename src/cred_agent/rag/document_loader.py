"""Document Loader for Cred Domain Support Agent Knowledge Base.

Track: Banking & FinTech (Cred)
Module: src.cred_agent.rag.document_loader

This module loads all Markdown policy documents from the knowledge base directory,
excluding README files, and returns structured document objects with metadata.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


# Mapping of document filename to domain topic name
DOCUMENT_TOPIC_MAP: dict = {
    "loan_eligibility.md": "loan eligibility criteria by loan type",
    "emi_calculation.md": "EMI calculation rules",
    "credit_card_fees.md": "credit-card fee structure",
    "kyc_requirements.md": "KYC document requirements",
    "fraud_disputes.md": "fraud-dispute resolution process",
    "account_closure.md": "account-closure process",
    "interest_rates.md": "interest-rate slabs",
    "prepayment_rules.md": "prepayment-penalty rules",
    "minimum_balance.md": "minimum-balance requirements",
    "credit_score.md": "credit-score impact factors",
    "joint_accounts.md": "joint-account rules",
    "nri_eligibility.md": "NRI-account eligibility",
}


@dataclass
class Document:
    """Structured representation of a loaded policy knowledge-base document."""

    doc_id: str
    doc_name: str
    filename: str
    source: str
    text: str
    raw_content: str
    metadata: dict = field(default_factory=dict)


def clean_markdown_text(raw_content: str) -> str:
    """Extract clean substantive text by removing markdown headers, blockquotes, and extra whitespace.

    Args:
        raw_content: Raw markdown string.

    Returns:
        Cleaned plain text suitable for chunking.
    """
    substantive_lines: List[str] = []
    for line in raw_content.splitlines():
        line_stripped = line.strip()
        # Exclude empty lines, markdown headers, and alert blockquotes
        if not line_stripped or line_stripped.startswith("#") or line_stripped.startswith(">"):
            continue
        substantive_lines.append(line_stripped)

    # Join and normalize consecutive whitespace
    cleaned = " ".join(substantive_lines)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_document_name(raw_content: str, fallback_stem: str) -> str:
    """Extract human-readable document name from the top markdown heading or filename.

    Args:
        raw_content: Raw markdown text.
        fallback_stem: Stem of the filename if no heading is found.

    Returns:
        Document name string.
    """
    match = re.search(r"^#\s+(?:Simulated Policy:\s*)?(.+)$", raw_content, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return fallback_stem.replace("_", " ").title()


def load_single_document(file_path: Union[str, Path]) -> Optional[Document]:
    """Load a single markdown file into a structured Document object.

    Args:
        file_path: Path to the .md file.

    Returns:
        Document object if the file is valid and non-empty, or None if empty.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file is not a markdown file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy file not found: {path.resolve()}")

    if path.suffix.lower() != ".md":
        raise ValueError(f"Expected a .md file, got: {path.name}")

    try:
        raw_content = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error(f"Failed to read file {path}: {exc}")
        raise

    # Handle empty or whitespace-only documents gracefully
    if not raw_content.strip():
        logger.warning(f"Skipping empty or whitespace-only document: {path.name}")
        return None

    cleaned_text = clean_markdown_text(raw_content)
    if not cleaned_text:
        logger.warning(f"Document {path.name} contains no substantive body text after cleaning.")
        return None

    doc_id = path.stem
    doc_name = extract_document_name(raw_content, fallback_stem=path.stem)
    filename = path.name
    source = str(path.as_posix())

    return Document(
        doc_id=doc_id,
        doc_name=doc_name,
        filename=filename,
        source=source,
        text=cleaned_text,
        raw_content=raw_content,
        metadata={"char_count": len(cleaned_text), "file_size_bytes": path.stat().st_size},
    )


def load_knowledge_base_documents(
    kb_dir: Union[str, Path] = Path("data/knowledge_base"),
) -> List[Document]:
    """Discover and load all non-README markdown documents from the knowledge-base directory.

    Args:
        kb_dir: Directory path containing the knowledge-base markdown documents.

    Returns:
        List of structured Document objects sorted deterministically by doc_id.

    Raises:
        FileNotFoundError: If the specified kb_dir does not exist.
        NotADirectoryError: If kb_dir is not a directory.
    """
    directory = Path(kb_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Knowledge-base directory does not exist: {directory.resolve()}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {directory.resolve()}")

    documents: List[Document] = []

    # Find all .md files in the directory
    for path in sorted(directory.glob("*.md")):
        # Requirement: Exclude README.md
        if path.name.lower() == "readme.md":
            logger.debug(f"Excluding readme file: {path.name}")
            continue

        doc = load_single_document(path)
        if doc is not None:
            documents.append(doc)

    return documents


# Backward compatibility alias
PolicyDocument = Document

if __name__ == "__main__":
    docs = load_knowledge_base_documents()
    print(f"Successfully loaded {len(docs)} documents from knowledge base:")
    for d in docs:
        print(f"  * [{d.doc_id}] '{d.doc_name}' ({d.filename}) -> {len(d.text)} characters")
