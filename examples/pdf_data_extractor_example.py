"""
Example of using the PDF Data Extractor to identify and extract
structured data from PDFs in parallel.
"""

import asyncio
from defog.llm.pdf_data_extractor import PDFDataExtractor
import json
import logging
from urllib.parse import urlparse
import re

logging.basicConfig(level=logging.INFO)


def generate_safe_filename(url: str) -> str:
    """
    Generate a safe filename from a URL for saving extracted data.

    Args:
        url: The URL to convert to a filename

    Returns:
        A safe filename string
    """
    try:
        parsed = urlparse(url)
        # Use the path component, removing leading slash and .pdf extension
        path_part = parsed.path.lstrip("/").replace(".pdf", "")
        # If path is empty, use the netloc (domain)
        if not path_part:
            path_part = parsed.netloc
        # Replace invalid filename characters with underscores
        safe_name = re.sub(r"[^\w\-_.]", "_", path_part)
        # Remove multiple consecutive underscores
        safe_name = re.sub(r"_+", "_", safe_name)
        # Remove leading/trailing underscores
        safe_name = safe_name.strip("_")
        # Ensure we have a non-empty filename
        if not safe_name:
            safe_name = "extracted_data"
        return f"extracted_data_{safe_name}.json"
    except Exception:
        # Fallback to a simple timestamp-based name if URL parsing fails
        import time

        timestamp = int(time.time())
        return f"extracted_data_{timestamp}.json"


async def extract(url):
    """Extract data from a PDF."""
    print(f"=== Data Extraction from {url} ===\n")

    # Create extractor
    extractor = PDFDataExtractor(
        analysis_model="claude-opus-4-20250514",
        extraction_model="claude-opus-4-20250514",
        max_parallel_extractions=10,
    )

    # Extract all data
    print("Analyzing PDF structure and extracting data...")
    result = await extractor.extract_all_data(pdf_url=url)

    print(f"\nDocument Type: {result.document_type}")
    print(f"Total datapoints identified: {result.total_datapoints_identified}")
    print(f"Successful extractions: {result.successful_extractions}")
    print(f"Failed extractions: {result.failed_extractions}")
    print(f"Total time: {result.total_time_ms / 1000:.2f} seconds")

    print(f"\n--- Cost Analysis ---")
    print(f"Total cost: ${result.total_cost_cents / 100:.4f}")
    print(
        f"Analysis cost (Step 1): ${result.metadata.get('analysis_cost_cents', 0.0) / 100:.4f}"
    )
    print(
        f"Extraction cost (Step 2+): ${result.metadata.get('extraction_cost_cents', 0.0) / 100:.4f}"
    )

    print(f"\n--- Token Usage ---")
    print(f"Total input tokens: {result.metadata.get('total_input_tokens', 0):,}")
    print(f"Total output tokens: {result.metadata.get('total_output_tokens', 0):,}")
    print(f"Total cached tokens: {result.metadata.get('total_cached_tokens', 0):,}")
    print(
        f"Total tokens: {result.metadata.get('total_input_tokens', 0) + result.metadata.get('total_output_tokens', 0):,}"
    )

    print("\n--- Extracted Datapoints ---")
    for extraction in result.extraction_results:
        if extraction.success:
            print(f"\n✅ {extraction.datapoint_name}:")
            print(
                f"   Cost: ${extraction.cost_cents / 100:.4f} | Tokens: {extraction.input_tokens + extraction.output_tokens:,} (in:{extraction.input_tokens:,}, out:{extraction.output_tokens:,}, cached:{extraction.cached_tokens:,})"
            )
        else:
            print(f"\n❌ {extraction.datapoint_name}: {extraction.error}")
            if extraction.cost_cents > 0:
                print(
                    f"   Cost: ${extraction.cost_cents / 100:.4f} | Tokens: {extraction.input_tokens + extraction.output_tokens:,}"
                )

    # save the extracted data to a json file
    filename = generate_safe_filename(url)
    with open(filename, "w") as f:
        json.dump(result.model_dump(), f, indent=2)


async def main():
    """Run all examples."""
    print("🚀 PDF Data Extractor Examples")
    print("=" * 60)

    try:
        # Run examples
        # Apple financial report (3 pages only)
        # await extract("https://www.apple.com/newsroom/pdfs/fy2025-q2/FY25_Q2_Consolidated_Financial_Statements.pdf")

        # Qwen 2.5 research paper (26 pages)
        # await extract("https://arxiv.org/pdf/2412.15115")

        # AI 2027 report
        await extract("https://ai-2027.com/ai-2027.pdf")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
