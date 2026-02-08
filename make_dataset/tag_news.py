import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from pydantic import BaseModel
from utils.openai import OpenAIRequest


class NewsTag(BaseModel):
    category: str
    sector: str


def return_prompt(headline: str, body: str) -> Tuple[str, str]:
    system_prompt = (
        "You are an economic expert specializing in forecasting the outlook of the U.S. financial markets."
    )

    prompt = []
    prompt += [
        "# Main Instruction",
        "You are given a single news ARTICLE. Your task is to analyze the scope of the economic impact of the ARTICLE and classify it into one of the following categories: Macro-Economic news, GICS sector-level news, or individual stock-level news.",
        "",
        "# Rules",
        "1. Each ARTICLE consists of a headline and a body.",
        "2. Please ignore the weird format of the ARTICLE",
        "3. The 11 GICS sectors are an industry classification system used to group U.S. stocks by industry, defined as follows:",
        "    3-1. Energy: The Energy sector consists of companies involved in the exploration, production, refining, and distribution of energy resources such as oil, natural gas, and coal. It includes oil & gas producers, integrated energy companies, and energy equipment and services providers. The sector is highly sensitive to commodity prices, geopolitical risks, and global economic conditions, and it often performs well during inflationary periods or early stages of economic recovery.",
        "    3-2. Materials: The Materials sector includes companies that produce raw materials such as metals, chemicals, construction materials, paper, and packaging products. Representative industries include steel, aluminum, chemicals, fertilizers, and cement. As these materials are essential inputs for industrial production, the sector is closely tied to the economic cycle and global manufacturing activity, particularly demand from emerging markets.",
        "    3-3. Industrials: The Industrials sector comprises companies engaged in manufacturing, construction, transportation, logistics, aerospace, defense, and electrical equipment. These firms form the backbone of the real economy and benefit from rising capital expenditures, infrastructure spending, and economic expansion. The sector typically performs strongly during periods of accelerating economic growth.",
        "    3-4. Consumer Discretionary: The Consumer Discretionary sector includes companies that provide non-essential goods and services whose demand depends on consumer income and confidence. This sector covers automobiles, apparel, luxury goods, hotels, leisure, restaurants, and e-commerce. It is highly sensitive to employment conditions and consumer sentiment, outperforming during economic booms but underperforming during downturns.",
        "    3-5. Consumer Staples: The Consumer Staples sector consists of companies that produce or distribute essential everyday products such as food, beverages, household goods, and personal care items. Major industries include food producers, beverage companies, tobacco firms, and large retailers. The sector is considered defensive, as demand remains relatively stable even during economic slowdowns.",
        "    3-6. Health Care: The Health Care sector includes pharmaceutical companies, biotechnology firms, medical device manufacturers, hospitals, and health care service providers. It benefits from long-term structural drivers such as aging populations and medical innovation, and is relatively resilient to economic cycles. However, regulatory changes, drug approvals, and clinical trial outcomes can significantly affect individual stocks.",
        "    3-7. Financials: The Financials sector comprises banks, insurance companies, investment banks, asset managers, and other financial service providers. The sector is highly influenced by interest rate levels and yield curve dynamics, with rising rates generally supporting bank profitability through higher net interest margins. Economic downturns can negatively impact the sector through increased credit losses.",
        "    3-8. Information Technology: The Information Technology sector includes companies involved in software, semiconductors, hardware, IT services, cloud computing, and artificial intelligence. It is characterized by high growth potential and innovation, serving as a key driver of productivity gains across the economy. While sensitive to interest rates due to valuation effects, it is widely viewed as a long-term structural growth sector.",
        "    3-9. Communication Services: The Communication Services sector includes telecommunications providers, media companies, entertainment firms, social media platforms, and internet content businesses. Formed by combining elements of the former telecom, technology, and consumer sectors, it is influenced by advertising cycles, content consumption trends, and platform competition. The sector exhibits a mix of growth and defensive characteristics.",
        "    3-10. Utilities: The Utilities sector consists of companies that provide essential public services such as electricity, gas, and water. It is known for stable cash flows and relatively high dividend yields, making it a defensive sector. However, it is sensitive to interest rate changes, as higher rates can reduce its relative attractiveness.",
        "    3-11. Real Estate: The Real Estate sector includes companies that own, develop, manage, and lease residential and commercial properties, as well as Real Estate Investment Trusts (REITs). Performance is driven by rental income, property values, and occupancy rates, and the sector is particularly sensitive to interest rates, financing conditions, and real estate market cycles.",
        "4. Determine the category of the news ARTICLE based on the following criteria:",
        "    4-1. Macro-Economic: News that analyzes or affects the overall U.S. economy.",
        "    4-2. GICS-Sector: News that does not impact the entire U.S. economy, but affects an entire GICS sector or multiple companies within a specific sector.",
        "    4-3. Stock: News that primarily affects a specific company or a small number of individual stocks, rather than the broader economy or a sector.",
        "    4-4: N/A: News that does not fit any of the above categories.",
        "5. If the category is GICS-Sector, identify which one of the 11 GICS sectors the news belongs to.",
        "",
        "# Output Format",
        "1. Example: {'category': ..., 'sector': ...}",
        "2. 'category' must be one of [Macro-Economic, GICS-Sector, Stock, N/A].",
        "3. 'sector' must be one of [Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate, 'N/A'].",
        "    3-1. If 'category' is Macro-Economic, Stock, or N/A, 'sector' must be N/A.",
        "    3-2. If 'category' is GICS-Sector, 'sector' must be one of [Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate].",
        "",
        "# Article",
        f"1. Headline: {headline}",
        f'2. Body: """\n{body}\n"""',
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


def _run_once(headline: str, body: str, model: str, max_retries: int) -> NewsTag:
    system_prompt, prompt = return_prompt(headline, body)
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
    result = _run_once(
        headline=news["articleTitle"],
        body=news["article"],
        model=args.model,
        max_retries=args.max_retries,
    )
    print({"category": result.category, "sector": result.sector})
