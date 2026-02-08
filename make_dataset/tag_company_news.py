import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from pydantic import BaseModel
from utils.openai import OpenAIRequest


class NewsTag(BaseModel):
    category: str


def return_prompt(
    headline: str,
    body: str,
    financial_statement: str,
    governance_risks: str,
    overview_product: str,
    recent_event_catalyst: str,
    strategy_market_ops: str,
) -> Tuple[str, str]:
    system_prompt = (
        "You are an economic expert specializing in forecasting the outlook of the U.S. financial markets."
    )

    prompt = []
    prompt += [
        "# Main Instruction",
        "You are given a single news article. Your task is to analyze the article and classify it into one of the following categories.",
        "",
        "# Descriptions of Categories",
        "1. Target-company news: News describing internal events that stem from the target company's own operations, decisions, or financial activities, as well as events that are directly aimed at or primarily affect the target company itself. To understand the target company’s current state when interpreting such events, we refer to the company profile.",
        "2. Related-company news: News describing external events that originate outside the target company and do not directly target it, but may indirectly influence the target company. Such external impacts may include developments involving competitor companies, partner firms, or suppliers. To understand the target company’s current state when interpreting these indirect effects, we refer to the company profile.",
        "3. N/A: The article does not fit any of the above categories.",
        "",
        "# COMPANY PROFILE OF TARGET COMPANY",
        f"1. Financial Statement: {financial_statement}",
        f"2. Governance Risks: {governance_risks}",
        f"3. Overview Product: {overview_product}",
        f"4. Recent Event Catalyst: {recent_event_catalyst}",
        f"5. Strategy Market Ops: {strategy_market_ops}",
        "",
        "# Rules",
        "1. Please ignore any unusual or inconsistent formatting in the article.",
        "",
        "# Output",
        '1. Example: {"category": ...}',
        '2. "category" must be one of Target-company news, Related-company news, or N/A.',
        "",
        "# Article",
        f"Headline: {headline}",
        f"Body: {body}",
    ]

    return system_prompt, "\n".join(prompt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single news tagging via OpenAI (no DB)."
    )
    parser.add_argument("--headline", type=str, default=None, help="News headline.")
    parser.add_argument("--body", type=str, default=None, help="News body text.")
    parser.add_argument(
        "--news-json",
        type=str,
        default=None,
        help="JSON string: {\"articleTitle\": ..., \"article\": ...}",
    )
    parser.add_argument(
        "--news-file",
        type=str,
        default=None,
        help="Path to JSON file with a news object.",
    )
    parser.add_argument(
        "--profile-json",
        type=str,
        default=None,
        help="JSON string with company profile fields.",
    )
    parser.add_argument(
        "--profile-file",
        type=str,
        default=None,
        help="Path to JSON file with company profile fields.",
    )
    parser.add_argument("--financial-statement", type=str, default="")
    parser.add_argument("--governance-risks", type=str, default="")
    parser.add_argument("--overview-product", type=str, default="")
    parser.add_argument("--recent-event-catalyst", type=str, default="")
    parser.add_argument("--strategy-market-ops", type=str, default="")
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="OpenAI model name."
    )
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def _load_news(args: argparse.Namespace) -> Dict:
    if args.news_json:
        return json.loads(args.news_json)
    if args.news_file:
        return json.loads(Path(args.news_file).read_text(encoding="utf-8"))
    if args.headline and args.body:
        return {"articleTitle": args.headline, "article": args.body}
    raise ValueError("Provide --news-json, --news-file, or both --headline and --body.")


def _load_profile(args: argparse.Namespace) -> Dict:
    if args.profile_json:
        return json.loads(args.profile_json)
    if args.profile_file:
        return json.loads(Path(args.profile_file).read_text(encoding="utf-8"))
    return {
        "financialStatement": args.financial_statement,
        "governanceRisks": args.governance_risks,
        "overviewProduct": args.overview_product,
        "recentEventCatalyst": args.recent_event_catalyst,
        "strategyMarketOps": args.strategy_market_ops,
    }


def _run_once(
    headline: str,
    body: str,
    profile: Dict,
    model: str,
    max_retries: int,
) -> NewsTag:
    system_prompt, prompt = return_prompt(
        headline=headline,
        body=body,
        financial_statement=profile.get("financialStatement", ""),
        governance_risks=profile.get("governanceRisks", ""),
        overview_product=profile.get("overviewProduct", ""),
        recent_event_catalyst=profile.get("recentEventCatalyst", ""),
        strategy_market_ops=profile.get("strategyMarketOps", ""),
    )
    last_error = None
    for _ in range(max_retries):
        try:
            response = OpenAIRequest.structured_request(
                system_prompt, prompt, NewsTag, model
            )
            return response
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


if __name__ == "__main__":
    args = _parse_args()
    news = _load_news(args)
    profile = _load_profile(args)
    result = _run_once(
        headline=news["articleTitle"],
        body=news["article"],
        profile=profile,
        model=args.model,
        max_retries=args.max_retries,
    )
    print({"category": result.category})
