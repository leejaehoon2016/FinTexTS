import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel
from utils.openai import OpenAIRequest


def return_prompt(
    ticker: str, news_elements: List[Dict], filing_doc: Dict
) -> Tuple[str, str]:
    system_prompt = (
        "You are an economic expert specializing in forecasting the outlook of the U.S. financial markets."
    )
    prompt = []
    prompt += [
        "# Main Instruction",
        f"You will be given a list of multiple ARTICLES that may impact {ticker}. ",
        f"Your task is to review all ARTICLES and identify up to three key categories of significant events that may impact {ticker}. ",
        f"For each category, summarize the key factual events in the related ARTICLES. ",
        f"",
        f"# Descriptions of {ticker}",
        f'1. Financial Statement: """\n{filing_doc.get("financialStatement", "N/A")}\n"""',
        f'2. Governance Risks: """\n{filing_doc.get("governanceRisks", "N/A")}\n"""',
        f'3. Overview Product: """\n{filing_doc.get("overviewProduct", "N/A")}\n"""',
        f'4. Recent Event Catalyst: """\n{filing_doc.get("recentEventCatalyst", "N/A")}\n"""',
        f'5. Strategy Market Ops: """\n{filing_doc.get("strategyMarketOps", "N/A")}\n"""',
        f"",
        f"# Rules",
        f"1. Please ignore any unusual or inconsistent formatting in the ARTICLES.",
        f"2. Select a category only if it has a meaningful impact on {ticker}.",
        f"3. If more than [N] categories are identified, select only the three most important ones.",
        f"4. Each category must address one single, distinct topic only.",
        f"5. Do not hallucinate. Write only based on the given ARTICLES.",
        "",
        "# Output Format",
        "1. Format: {",
         "    'category1': ...,",
         "    'category2': ...,",
         "    'category3': ...,",
         "}",
        "",
        "# Articles",
    ]
    for i, e in enumerate(news_elements, 1):
        headline, body = e["articleTitle"], e["article"]
        prompt += [
            f"{i}. ARTICLE {i}:",
            f"    {i}-1. Headline: {headline}",
            f'    {i}-2. Body: """\n{body}\n"""',
            f"",
        ]

    return system_prompt, "\n".join(prompt)


class NewsSummary(BaseModel):
    category1: str
    category2: str
    category3: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Company summary via OpenAI (multi news input)."
    )
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument(
        "--news-json",
        type=str,
        default=None,
        help="JSON string: [{\"articleTitle\": ..., \"article\": ...}, ...]",
    )
    parser.add_argument(
        "--news-file",
        type=str,
        default=None,
        help="Path to JSON file with list of news objects.",
    )
    parser.add_argument(
        "--filing-json",
        type=str,
        default=None,
        help="JSON string: {\"financialStatement\": ..., ...}",
    )
    parser.add_argument(
        "--filing-file",
        type=str,
        default=None,
        help="Path to JSON file with filing object.",
    )
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def _load_json_str(value: str):
    return json.loads(value) if value else None


def _load_json_file(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def _load_news_elements(args: argparse.Namespace) -> List[Dict]:
    news = _load_json_str(args.news_json) or _load_json_file(args.news_file)
    if not news:
        raise ValueError("Provide --news-json or --news-file.")
    return news


def _load_filing_doc(args: argparse.Namespace) -> Dict:
    filing = _load_json_str(args.filing_json) or _load_json_file(args.filing_file)
    if not filing:
        raise ValueError("Provide --filing-json or --filing-file.")
    return filing


def _run_once(
    ticker: str,
    news_elements: List[Dict],
    filing_doc: Dict,
    model: str,
    max_retries: int,
) -> NewsSummary:
    system_prompt, prompt = return_prompt(ticker, news_elements, filing_doc)
    last_error = None
    for _ in range(max_retries):
        try:
            response = OpenAIRequest.structured_request(
                system_prompt, prompt, NewsSummary, model
            )
            return response
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


if __name__ == "__main__":
    args = _parse_args()
    news_elements = _load_news_elements(args)
    filing_doc = _load_filing_doc(args)
    result = _run_once(
        ticker=args.ticker,
        news_elements=news_elements,
        filing_doc=filing_doc,
        model=args.model,
        max_retries=args.max_retries,
    )
    print(result.model_dump())
