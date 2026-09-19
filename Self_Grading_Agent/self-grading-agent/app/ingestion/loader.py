from pathlib import Path


def extract_metadata(content: str) -> dict[str, str]:
    """Extract standard metadata fields from a Markdown document."""

    metadata = {}

    for line in content.splitlines():
        line = line.strip()

        if line.startswith("Document ID:"):
            metadata["document_id"] = line.split(":", 1)[1].strip()

        elif line.startswith("Document Type:"):
            metadata["document_type"] = line.split(":", 1)[1].strip()

        elif line.startswith("Last Updated:"):
            metadata["last_updated"] = line.split(":", 1)[1].strip()

    return metadata


def load_documents(knowledge_base_path: Path) -> list[dict]:
    """Load Markdown documents and preserve their content and metadata."""

    if not knowledge_base_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {knowledge_base_path}"
        )

    documents = []

    for file_path in sorted(knowledge_base_path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        metadata = extract_metadata(content)

        documents.append(
            {
                "document_id": metadata.get(
                    "document_id", file_path.stem
                ),
                "document_name": file_path.name,
                "document_type": metadata.get("document_type", "Unknown"),
                "last_updated": metadata.get("last_updated", "Unknown"),
                "content": content,
            }
        )

    if not documents:
        raise ValueError(
            f"No Markdown documents found in: {knowledge_base_path}"
        )

    return documents