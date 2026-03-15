from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from projects.shikiho_text_parser.paths import OUTPUT_DIR

NOISE_PATTERNS = [
    "会社四季報オンライン",
    "株式投資・銘柄研究のバイブル",
    "四季報先取り",
    "トップ",
    "記事",
    "銘柄研究",
    "スクリーニング",
    "業界研究",
    "米国株",
    "四季報AI",
    "チャート",
    "ウォッチ",
    "簡易チャート",
    "詳細チャート",
    "業績財務の詳細を見る",
    "配当を見る",
    "もっと見る",
    "株主の詳細",
    "役員の詳細",
]

HEADINGS = [
    "特色",
    "連結事業",
    "四季報スコア",
    "業績",
    "業績予想更新",
    "株価指標",
    "適時開示情報",
    "採用",
    "株式",
    "株主",
    "役員",
    "連結",
    "財務",
    "指標等",
]


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = raw.replace("\u3000", " ").strip()
        if not line:
            continue
        if any(line == pat or line.startswith(pat) for pat in NOISE_PATTERNS):
            continue
        if line in {"会社HP", "最新の四季報", "プロフィール", "四半期業績", "長期業績", "株主優待", "株価推移", "誌面アーカイブ"}:
            continue
        if line.startswith("http://") or line.startswith("https://"):
            continue
        lines.append(line)
    return lines


def find_heading_index(lines: list[str], heading: str) -> int | None:
    for i, line in enumerate(lines):
        if line == heading:
            return i
    return None


def extract_section(lines: list[str], heading: str) -> list[str]:
    start = find_heading_index(lines, heading)
    if start is None:
        return []
    next_indices = [find_heading_index(lines, h) for h in HEADINGS if find_heading_index(lines, h) is not None and find_heading_index(lines, h) > start]
    end = min(next_indices) if next_indices else len(lines)
    return lines[start + 1:end]


def extract_basic_info(lines: list[str]) -> dict:
    info = {
        "ticker_code": "",
        "market": "",
        "company_name": "",
        "disclosure_date": "",
        "flags": [],
        "categories": [],
    }
    code_idx = None
    for i, line in enumerate(lines[:30]):
        m = re.match(r"^([0-9]{4}[A-Z]?)\s+(.+)$", line)
        if m:
            info["ticker_code"] = m.group(1)
            info["market"] = m.group(2).strip()
            code_idx = i
            break
    if code_idx is None:
        return info
    if code_idx + 2 < len(lines):
        info["company_name"] = lines[code_idx + 2]
    for i in range(code_idx, min(code_idx + 10, len(lines))):
        line = lines[i]
        if line.startswith("直近決算発表日："):
            info["disclosure_date"] = line.split("：", 1)[1].strip()
    meta_pool = []
    for line in lines[code_idx + 3:]:
        if line == "特色":
            break
        meta_pool.append(line)
    flags = []
    categories = []
    for line in meta_pool:
        if any(key in line for key in ["あり", "貸借", "信用", "注意"]):
            flags.append(line)
        else:
            categories.append(line)
    info["flags"] = flags
    info["categories"] = categories[:5]
    return info


def extract_shikiho_scores(lines: list[str]) -> dict:
    section = extract_section(lines, "四季報スコア")
    scores = {"overall": None}
    if not section:
        return scores
    for line in section:
        if re.fullmatch(r"[0-9]+", line):
            scores["overall"] = int(line)
            break
    keys = ["成長性", "収益性", "安全性", "規模", "割安度", "値上がり"]
    for key in keys:
        for i, line in enumerate(section):
            if line == key and i + 1 < len(section):
                value_line = section[i + 1]
                m = re.search(r"([0-9]+)", value_line)
                if m:
                    scores[key] = int(m.group(1))
                break
    return scores


def extract_headline_blocks(lines: list[str]) -> list[dict]:
    blocks = []
    for line in lines:
        if line.startswith("【") and "】" in line:
            title, body = line.split("】", 1)
            blocks.append({"title": title + "】", "body": body.strip(" \t")})
    return blocks


def parse_table_rows(section_lines: list[str]) -> list[dict]:
    rows = []
    for line in section_lines:
        if re.match(r"^(連|単|会)[0-9]{2}\.", line):
            parts = re.split(r"\s+", line)
            rows.append(
                {
                    "period": parts[0],
                    "values": parts[1:],
                    "raw": line,
                }
            )
    return rows


def extract_stock_indicators(lines: list[str]) -> dict:
    section = extract_section(lines, "株価指標")
    text = "\n".join(section)
    out = {
        "market_cap": "",
        "min_purchase_amount": "",
        "unit_shares": "",
        "trading_value": "",
        "volume": "",
        "forecast_per": [],
        "actual_pbr": "",
        "forecast_dividend_yield": [],
        "ytd_return": "",
        "ma200_gap": "",
    }
    for line in section:
        if "時価総額" in line:
            out["market_cap"] = line.replace("時価総額", "").strip()
        elif "最低購入金額" in line:
            out["min_purchase_amount"] = line.replace("最低購入金額", "").strip()
        elif "売買単位" in line:
            out["unit_shares"] = line.replace("売買単位", "").strip()
        elif line.startswith("売買代金"):
            out["trading_value"] = line.replace("売買代金", "").strip()
        elif line.startswith("出来高"):
            out["volume"] = line.replace("出来高", "").strip()
        elif "実績PBR" in line:
            continue
        elif "年初来株価上昇率" in line:
            out["ytd_return"] = line.replace("年初来株価上昇率", "").strip()
        elif "200日移動平均乖離率" in line:
            out["ma200_gap"] = line.replace("200日移動平均乖離率", "").strip()
    out["forecast_per"] = re.findall(r"連[0-9]{2}\.[0-9]+([0-9.]+)倍", text)
    pbr_match = re.search(r"実績PBR\s+([0-9.]+倍)", text)
    if pbr_match:
        out["actual_pbr"] = pbr_match.group(1)
    out["forecast_dividend_yield"] = re.findall(r"連[0-9]{2}\.[0-9]+([0-9.]+%)", text)
    return out


def extract_company_profile(lines: list[str]) -> dict:
    keys = ["上場", "設立", "本社", "従業員", "証券", "銀行", "監査", "仕入先", "販売先"]
    out = {}
    for i, line in enumerate(lines):
        if line in keys and i + 1 < len(lines):
            out[line] = lines[i + 1]
        elif any(line.startswith(f"{key} ") for key in keys):
            key, value = line.split(" ", 1)
            out[key] = value.strip()
    return out


def extract_shareholders(lines: list[str]) -> list[str]:
    section = extract_section(lines, "株主")
    holders = []
    for line in section:
        if re.search(r"\([0-9.]+%\)", line):
            holders.append(line)
    return holders[:10]


def extract_financials(lines: list[str]) -> dict:
    section = extract_section(lines, "財務")
    out = {}
    for line in section:
        m = re.match(r"^(総資産|自己資本|自己資本比率|資本金|利益剰余金|有利子負債)\s+(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_shikiho_text(text: str, source_name: str) -> dict:
    lines = clean_lines(text)
    result = {
        "source_file": source_name,
        "basic_info": extract_basic_info(lines),
        "feature_summary": " ".join(extract_section(lines, "特色")[:1]),
        "segment_summary": " ".join(extract_section(lines, "連結事業")[:2]),
        "shikiho_scores": extract_shikiho_scores(lines),
        "headline_blocks": extract_headline_blocks(lines),
        "earnings_rows": parse_table_rows(extract_section(lines, "業績")),
        "guidance_rows": parse_table_rows(extract_section(lines, "業績予想更新")),
        "stock_indicators": extract_stock_indicators(lines),
        "company_profile": extract_company_profile(lines),
        "shareholders": extract_shareholders(lines),
        "financials": extract_financials(lines),
    }
    return result


def flatten_summary(parsed: dict) -> dict:
    basic = parsed["basic_info"]
    score = parsed["shikiho_scores"]
    indicators = parsed["stock_indicators"]
    financials = parsed["financials"]
    return {
        "source_file": parsed["source_file"],
        "ticker_code": basic.get("ticker_code", ""),
        "company_name": basic.get("company_name", ""),
        "market": basic.get("market", ""),
        "disclosure_date": basic.get("disclosure_date", ""),
        "flags": " / ".join(basic.get("flags", [])),
        "categories": " / ".join(basic.get("categories", [])),
        "feature_summary": parsed.get("feature_summary", ""),
        "segment_summary": parsed.get("segment_summary", ""),
        "shikiho_score_overall": score.get("overall"),
        "score_growth": score.get("成長性"),
        "score_profitability": score.get("収益性"),
        "score_safety": score.get("安全性"),
        "score_scale": score.get("規模"),
        "score_undervalued": score.get("割安度"),
        "score_momentum": score.get("値上がり"),
        "headline_count": len(parsed.get("headline_blocks", [])),
        "earnings_row_count": len(parsed.get("earnings_rows", [])),
        "guidance_row_count": len(parsed.get("guidance_rows", [])),
        "market_cap": indicators.get("market_cap", ""),
        "min_purchase_amount": indicators.get("min_purchase_amount", ""),
        "volume": indicators.get("volume", ""),
        "actual_pbr": indicators.get("actual_pbr", ""),
        "ytd_return": indicators.get("ytd_return", ""),
        "ma200_gap": indicators.get("ma200_gap", ""),
        "equity_ratio": financials.get("自己資本比率", ""),
        "interest_bearing_debt": financials.get("有利子負債", ""),
    }


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.txt"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse copied Shikiho text into structured JSON/CSV.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = args.output_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for txt_path in collect_input_files(args.input):
        raw = txt_path.read_text(encoding="utf-8")
        parsed = parse_shikiho_text(raw, txt_path.name)
        json_path = parsed_dir / f"{txt_path.stem}.json"
        json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(flatten_summary(parsed))

    summary_csv = args.output_dir / "parsed_summary.csv"
    if summaries:
        with summary_csv.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    print(f"[OUT] parsed_dir={parsed_dir}")
    print(f"[OUT] summary={summary_csv}")
    print(f"[INFO] parsed_files={len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
