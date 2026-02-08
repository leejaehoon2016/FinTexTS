import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel
from utils.openai import OpenAIRequest


def return_prompt(tag: str, date: str, news_elements: List[Dict]) -> Tuple[str, str]:
    tag2desc = {
        "Macro-Economic": "the overall U.S. economy.",
        "Energy":         "the Energy sector (The Energy sector consists of companies involved in the exploration, production, refining, and distribution of energy resources such as oil, natural gas, and coal. It includes oil & gas producers, integrated energy companies, and energy equipment and services providers. The sector is highly sensitive to commodity prices, geopolitical risks, and global economic conditions, and it often performs well during inflationary periods or early stages of economic recovery.)",
        "Materials":      "the Materials sector (The Materials sector includes companies that produce raw materials such as metals, chemicals, construction materials, paper, and packaging products. Representative industries include steel, aluminum, chemicals, fertilizers, and cement. As these materials are essential inputs for industrial production, the sector is closely tied to the economic cycle and global manufacturing activity, particularly demand from emerging markets.)",
        "Industrials":    "the Industrials sector (The Industrials sector comprises companies engaged in manufacturing, construction, transportation, logistics, aerospace, defense, and electrical equipment. These firms form the backbone of the real economy and benefit from rising capital expenditures, infrastructure spending, and economic expansion. The sector typically performs strongly during periods of accelerating economic growth.)",
        "Consumer Discretionary": "the Consumer Discretionary sector (The Consumer Discretionary sector includes companies that provide non-essential goods and services whose demand depends on consumer income and confidence. This sector covers automobiles, apparel, luxury goods, hotels, leisure, restaurants, and e-commerce. It is highly sensitive to employment conditions and consumer sentiment, outperforming during economic booms but underperforming during downturns.)",
        "Consumer Staples": "the Consumer Staples sector (The Consumer Staples sector consists of companies that produce or distribute essential everyday products such as food, beverages, household goods, and personal care items. Major industries include food producers, beverage companies, tobacco firms, and large retailers. The sector is considered defensive, as demand remains relatively stable even during economic slowdowns.)",
        "Health Care":      "the Health Care sector (The Health Care sector includes pharmaceutical companies, biotechnology firms, medical device manufacturers, hospitals, and health care service providers. It benefits from long-term structural drivers such as aging populations and medical innovation, and is relatively resilient to economic cycles. However, regulatory changes, drug approvals, and clinical trial outcomes can significantly affect individual stocks.)",
        "Financials":       "the Financials sector (The Financials sector comprises banks, insurance companies, investment banks, asset managers, and other financial service providers. The sector is highly influenced by interest rate levels and yield curve dynamics, with rising rates generally supporting bank profitability through higher net interest margins. Economic downturns can negatively impact the sector through increased credit losses.)",
        "Information Technology": "the Information Technology sector (The Information Technology sector includes companies involved in software, semiconductors, hardware, IT services, cloud computing, and artificial intelligence. It is characterized by high growth potential and innovation, serving as a key driver of productivity gains across the economy. While sensitive to interest rates due to valuation effects, it is widely viewed as a long-term structural growth sector.)",
        "Communication Services": "the Communication Services sector (The Communication Services sector includes telecommunications providers, media companies, entertainment firms, social media platforms, and internet content businesses. Formed by combining elements of the former telecom, technology, and consumer sectors, it is influenced by advertising cycles, content consumption trends, and platform competition. The sector exhibits a mix of growth and defensive characteristics.)",
        "Utilities":        "the Utilities sector (The Utilities sector consists of companies that provide essential public services such as electricity, gas, and water. It is known for stable cash flows and relatively high dividend yields, making it a defensive sector. However, it is sensitive to interest rate changes, as higher rates can reduce its relative attractiveness.)",
        "Real Estate":      "the Real Estate sector (The Real Estate sector includes companies that own, develop, manage, and lease residential and commercial properties, as well as Real Estate Investment Trusts (REITs). Performance is driven by rental income, property values, and occupancy rates, and the sector is particularly sensitive to interest rates, financing conditions, and real estate market cycles.)",
    }
    system_prompt = (
        "You are an economic expert specializing in forecasting the outlook of the U.S. financial markets."
    )

    prompt = []

    prompt += [
        "# Main Instruction",
        f"You will be given a list of multiple ARTICLES that may impact {tag}. ",
        f"Your task is to review all ARTICLES and identify up to five key categories of significant events that may impact {tag}. ",
        f"For each category, summarize the key factual events in the related ARTICLES. ",
        f"",
        f"# Descriptions of {tag}",
        f"{tag2desc[tag]}",
        f"",
        f"# Rules",
        f"1. Please ignore any unusual or inconsistent formatting in the ARTICLES.",
        f"2. Select a category only if it has a meaningful impact on {tag}.",
        f"3. If more than [N] categories are identified, select only the five most important ones.",
        f"4. Each category must address one single, distinct topic only.",
        f"5. Do not hallucinate. Write only based on the given ARTICLES.",
        "",
        "# Output Format",
        "1. Format: {",
         "    'category1': ...,",
         "    'category2': ...,",
         "    'category3': ...,",
         "    'category4': ...,",
         "    'category5': ...,",
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
    category4: str
    category5: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Macro/Sector summary via OpenAI (multi news input)."
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        choices=[
            "Macro-Economic",
            "Energy",
            "Materials",
            "Industrials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Health Care",
            "Financials",
            "Information Technology",
            "Communication Services",
            "Utilities",
            "Real Estate",
        ],
        help="Target tag (macro or sector).",
    )
    parser.add_argument("--headline", type=str, default=None)
    parser.add_argument("--body", type=str, default=None)
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
    parser.add_argument("--date", type=str, default="N/A")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


def _load_news_elements(args: argparse.Namespace) -> List[Dict]:
    if args.news_json:
        return json.loads(args.news_json)
    if args.news_file:
        news_path = Path(args.news_file)
        return json.loads(news_path.read_text(encoding="utf-8"))
    if args.headline and args.body:
        return [{"articleTitle": args.headline, "article": args.body}]
    raise ValueError("Provide --news-json, --news-file, or both --headline and --body.")


def _run_once(
    tag: str,
    date: str,
    news_elements: List[Dict],
    model: str,
    max_retries: int,
) -> NewsSummary:
    system_prompt, prompt = return_prompt(tag, date, news_elements)
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
    result = _run_once(
        tag=args.tag,
        date=args.date,
        news_elements=news_elements,
        model=args.model,
        max_retries=args.max_retries,
    )
    print(
        {
            "category1": result.category1,
            "category2": result.category2,
            "category3": result.category3,
            "category4": result.category4,
            "category5": result.category5,
        }
    )


