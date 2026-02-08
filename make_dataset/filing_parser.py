import argparse
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel, ConfigDict
from utils.openai import OpenAIRequest


def _build_prompt(sec_filing_text: str) -> Tuple[str, str]:
    system_prompt = (
        "You are an economic expert specializing in forecasting the outlook of the U.S. financial markets."
    )
    prompt = [
        "# Main Instruction",
        "You are given a single SEC filing document. Your task is to identify and summarize all relevant content in the SEC filing",
        "according to the following five categories. Extract and summarize only the information that is relevant to each category.",
        "",
        "# Descriptions of the 5 Categories",
        "1. overviewProduct: Provides a high-level summary of the company's business, including its mission, core products or services, key customer groups, business segments, and primary geographic markets.",
        "2. strategyMarketOps: Describes the company's strategic direction and competitive strengths (e.g., proprietary technology, intellectual property, and regulatory expertise), along with its target markets, regulatory context, and operating model such as manufacturing footprint, supply chain structure, and major partnerships.",
        "3. governanceRisks: Covers the company's governance framework, including notable changes in leadership or the board, and summarizes key risks disclosed in filings. Risk factors are organized by category (e.g., regulatory, market, operational, and cybersecurity) and may include both long-term structural risks and near-term concerns.",
        "4. financialStatement: Summarizes the company's financial statements with key figures, and provides an interpretation of its financial health based on filings-discussing major performance drivers, liquidity, funding and capital resources, capital allocation decisions, accounting updates, and material obligations-rather.",
        "5. recentEventCatalyst: highlights significant developments from roughly the past 12 months, including changes to earnings outlook, major product releases, regulatory decisions, M&A progress, leadership updates, and other events that could meaningfully affect market perception or performance.",
        "",
        "# Rules",
        "1. The SEC filing can be long and may contain noisy formatting. Focus on the content rather than the format.",
        "2. Do not hallucinate. Only use information explicitly stated in the SEC filing.",
        "3. If there is no relevant information for a category, return an empty string for that category.",
        "",
        "# Output Format",
        "1. Example: {",
        '"overviewProduct": ...,',
        '"strategyMarketOps": ...,',
        '"financialStatement": ...,',
        '"governanceRisks": ...,',
        '"recentEventCatalyst": ...',
        "}",
        "# SEC Filing",
        sec_filing_text,
    ]
    return system_prompt, "\n".join(prompt)


class FilingSummary(BaseModel):
    overviewProduct: str
    strategyMarketOps: str
    financialStatement: str
    governanceRisks: str
    recentEventCatalyst: str
    model_config = ConfigDict(extra="forbid")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a single SEC filing.")
    parser.add_argument("--text", type=str, default=None, help="SEC filing text.")
    parser.add_argument("--file", type=str, default=None, help="Path to SEC filing text file.")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def _load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    raise ValueError("Provide --text or --file.")


def _run_once(sec_filing_text: str, model: str, max_retries: int) -> FilingSummary:
    system_prompt, prompt = _build_prompt(sec_filing_text)
    last_error = None
    for _ in range(max_retries):
        try:
            response = OpenAIRequest.structured_request(
                system_prompt, prompt, FilingSummary, model
            )
            return response
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


if __name__ == "__main__":
    args = _parse_args()
    text = _load_text(args)
    result = _run_once(text, args.model, args.max_retries)
    print(result.model_dump())
