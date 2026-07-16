"""四季報headlinesコメントをsentence-transformersでベクトル化し、shikiho_embeddingsに格納する。

使用例:
    python scripts/generate_shikiho_embeddings.py
    python scripts/generate_shikiho_embeddings.py --model intfloat/multilingual-e5-small
    python scripts/generate_shikiho_embeddings.py --issue-code 2026-1 --limit 20 --dry-run
"""
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]

# プロキシ環境: HuggingFace Hubからのモデルダウンロードに必要
_CERT_PATH = str(PROJECT_ROOT / "win_certs.pem")
os.environ.setdefault("SSL_CERT_FILE", _CERT_PATH)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CERT_PATH)

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DB_DSN = "postgresql://postgres:ogm384@localhost:5432/stock_selection"

# 検索文書として埋め込むためのプレフィックス（モデルごとの規約に合わせる）
DOC_PREFIXES = {
    "cl-nagoya/ruri-v3-310m": "検索文書: ",
    "intfloat/multilingual-e5-small": "passage: ",
}


def fetch_headline_texts(conn, issue_code: str, limit: int | None) -> list[tuple[str, str]]:
    """ticker_codeごとにtitle+bodyをseq順に連結して返す。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ticker_code,
                   string_agg(title || body, '' ORDER BY seq) AS source_text
            FROM shikiho_headlines
            WHERE issue_code = %s
            GROUP BY ticker_code
            ORDER BY ticker_code
            """,
            (issue_code,),
        )
        rows = cur.fetchall()
    if limit:
        rows = rows[:limit]
    return rows


def upsert_embeddings(
    conn,
    issue_code: str,
    kind: str,
    model_name: str,
    ticker_codes: list[str],
    source_texts: list[str],
    embeddings,
) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO shikiho_embeddings
                (ticker_code, issue_code, kind, model, embedding, source_text)
            VALUES %s
            ON CONFLICT (ticker_code, issue_code, kind, model)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                source_text = EXCLUDED.source_text,
                created_at = now()
            """,
            [
                (ticker, issue_code, kind, model_name, emb.tolist(), text)
                for ticker, text, emb in zip(ticker_codes, source_texts, embeddings)
            ],
        )
    conn.commit()


def run_similarity_check(conn, issue_code: str, kind: str, model_name: str) -> None:
    """任意の1銘柄を選び、コサイン距離で類似銘柄TOP5を表示する。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.ticker_code, r.symbol, r.company_name
            FROM shikiho_embeddings e
            JOIN shikiho_reports r
              ON r.ticker_code = e.ticker_code AND r.issue_code = e.issue_code
            WHERE e.issue_code = %s AND e.kind = %s AND e.model = %s
            ORDER BY e.ticker_code
            LIMIT 1
            """,
            (issue_code, kind, model_name),
        )
        seed = cur.fetchone()
        if seed is None:
            print("類似度確認用の埋め込みが見つかりませんでした。")
            return
        seed_ticker, seed_symbol, seed_name = seed
        print(f"\n=== 類似銘柄検索テスト: {seed_ticker} ({seed_symbol}) {seed_name} ===")

        cur.execute(
            """
            SELECT e2.ticker_code, r2.symbol, r2.company_name,
                   e1.embedding <=> e2.embedding AS cosine_distance
            FROM shikiho_embeddings e1
            JOIN shikiho_embeddings e2
              ON e2.issue_code = e1.issue_code AND e2.kind = e1.kind AND e2.model = e1.model
              AND e2.ticker_code != e1.ticker_code
            JOIN shikiho_reports r2
              ON r2.ticker_code = e2.ticker_code AND r2.issue_code = e2.issue_code
            WHERE e1.ticker_code = %s AND e1.issue_code = %s AND e1.kind = %s AND e1.model = %s
            ORDER BY cosine_distance ASC
            LIMIT 5
            """,
            (seed_ticker, issue_code, kind, model_name),
        )
        for ticker, symbol, name, dist in cur.fetchall():
            print(f"  {ticker} ({symbol}) {name}: distance={dist:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="cl-nagoya/ruri-v3-310m",
        help="sentence-transformersモデル名 (既定: cl-nagoya/ruri-v3-310m)",
    )
    parser.add_argument("--issue-code", default="2026-1", help="対象issue_code (既定: 2026-1)")
    parser.add_argument("--kind", default="headlines", help="embeddingの種別 (既定: headlines)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="デバッグ用: 件数を絞る")
    parser.add_argument("--dry-run", action="store_true", help="DBへの書き込みを行わない")
    parser.add_argument(
        "--skip-check", action="store_true", help="格納後の類似銘柄検索テストをスキップ"
    )
    args = parser.parse_args()

    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)

    print(f"headlines取得中... issue_code={args.issue_code}")
    rows = fetch_headline_texts(conn, args.issue_code, args.limit)
    if not rows:
        print("対象データがありません。")
        return
    ticker_codes = [r[0] for r in rows]
    source_texts = [r[1] for r in rows]
    print(f"{len(ticker_codes)}銘柄分のテキストを取得しました。")

    print(f"モデル読み込み中: {args.model}")
    model = SentenceTransformer(args.model)

    prefix = DOC_PREFIXES.get(args.model, "")
    encode_texts = [prefix + t for t in source_texts]

    print("ベクトル化中...")
    embeddings = model.encode(
        encode_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"埋め込み次元数: {embeddings.shape[1]}")

    if args.dry_run:
        print("dry-runのためDB書き込みはスキップします。")
    else:
        print("shikiho_embeddingsへ格納中...")
        upsert_embeddings(
            conn, args.issue_code, args.kind, args.model, ticker_codes, source_texts, embeddings
        )
        print("格納完了。")

        if not args.skip_check:
            run_similarity_check(conn, args.issue_code, args.kind, args.model)

    conn.close()


if __name__ == "__main__":
    main()
