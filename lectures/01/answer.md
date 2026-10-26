# 第1回 解答

## 演習

- 講座に固有の質問（宿題の3レベル）へのモデルの回答（要約でよい）：抽象的かつ一般的なレベル3の説明
- 用語集の正しい意味と比べて気づいたこと：より具体的でわかりやすい説明
- 表示できたトークン数（入力 / 出力）：7/630

## 宿題

- 取り組んだレベル：発展
- やったこと：発展課題に取り組み、LLMの応答時間と入出力トークン数を計測する機能を実装した。また、複数のモデル・質問に対して繰り返し実験を行い、結果をCSV形式で保存できるプログラムを作成した。さらに、無料枠のレート制限を考慮し、APIの呼び出し間隔の調整やエラー処理を追加した。
- 結果：実際のAPI呼び出しでは、応答時間2.79秒、入力7トークン、出力489トークン、合計496トークンと計測できた。また、比較実験用のプログラムについても、複数条件の記録やCSV出力が可能であることを確認した。
- 気づき・疑問：応答時間には通信時間も含まれるため、単純にモデルの処理性能を示すわけではないことに気づいた。また、思考トークン数が取得できない場合もあり、モデルによってトークン消費の仕組みが異なる点が興味深かった。今後は、モデルの性能とトークン消費量の関係や、無料枠のレート制限が実用上どの程度影響するのかを検証したい。
- 追加（改変）したコード

"""第1回: 素のLLMに質問し、応答時間とトークン数を表示する。"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())

    start = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata
    print(f"\n[応答時間] {elapsed:.2f}秒")
    if usage is not None:
        print(f"[入力] {usage.prompt_token_count} tokens")
        print(f"[出力] {usage.candidates_token_count} tokens")
        print(f"[思考] {usage.thoughts_token_count} tokens")
        print(f"[合計] {usage.total_token_count} tokens")

    return response.text or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()



"""第1回・発展: Geminiモデル別の応答時間とトークン消費をCSVに記録する。"""

import argparse
import csv
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTIONS = [
    "LoRAとは何ですか？150字以内で説明してください。",
    "AIE2026の講座の宿題にある『3レベル』とは何ですか？150字以内で説明してください。",
]

FIELDS = [
    "timestamp_utc", "model", "question", "repeat", "status", "elapsed_sec",
    "input_tokens", "output_tokens", "thought_tokens", "total_tokens",
    "error_code", "answer",
]


def run_one(client, model: str, question: str, repeat: int) -> dict:
    """1回呼び出し、成功・失敗にかかわらず計測結果を返す。"""
    row = dict.fromkeys(FIELDS, "")
    row.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "question": question, "repeat": repeat,
    })
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=question)
        usage = response.usage_metadata
        row["status"] = "ok"
        row["answer"] = response.text or ""
        if usage is not None:
            row["input_tokens"] = usage.prompt_token_count
            row["output_tokens"] = usage.candidates_token_count
            row["thought_tokens"] = usage.thoughts_token_count
            row["total_tokens"] = usage.total_token_count
    except errors.APIError as e:
        row["status"] = "rate_limited" if e.code == 429 else "api_error"
        row["error_code"] = e.code
        row["answer"] = api_error_message(e.code, e.message)
    finally:
        row["elapsed_sec"] = round(time.perf_counter() - start, 3)
    return row


def print_summary(rows: list[dict]) -> None:
    print("\n--- モデル別の集計 ---")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    for model, items in grouped.items():
        ok = [r for r in items if r["status"] == "ok"]
        limited = sum(r["status"] == "rate_limited" for r in items)
        other_errors = len(items) - len(ok) - limited
        print(f"\n{model}: 成功 {len(ok)}/{len(items)}, 429={limited}, その他エラー={other_errors}")
        if not ok:
            continue
        avg_sec = mean(float(r["elapsed_sec"]) for r in ok)
        print(f"  平均応答時間: {avg_sec:.2f}秒")
        for field, label in [
            ("input_tokens", "入力"),
            ("output_tokens", "出力"),
            ("thought_tokens", "思考"),
            ("total_tokens", "合計"),
        ]:
            values = [int(r[field]) for r in ok if r[field] not in (None, "")]
            if values:
                print(f"  平均{label}トークン: {mean(values):.1f} (計 {sum(values)})")
            else:
                print(f"  平均{label}トークン: データなし")

    print("\n--- モデル × 質問 別の平均 ---")
    by_condition = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_condition[(row["model"], row["question"])].append(row)
    for (model, question), items in by_condition.items():
        avg_sec = mean(float(r["elapsed_sec"]) for r in items)
        totals = [int(r["total_tokens"]) for r in items
                  if r["total_tokens"] not in (None, "")]
        avg_total = f"{mean(totals):.1f}" if totals else "データなし"
        print(f"{model} / {question}")
        print(f"  n={len(items)} / 平均応答時間={avg_sec:.2f}秒 / 平均合計トークン={avg_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Geminiモデルの比較実験")
    parser.add_argument("--models", nargs="+", default=[MODEL], help="比較するモデル名")
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS, help="比較する質問")
    parser.add_argument("--repeat", type=int, default=3, help="各条件の実行回数")
    parser.add_argument("--interval", type=float, default=15.0, help="APIリクエスト間の待機秒数")
    parser.add_argument("--csv", default="benchmark_results.csv", help="結果の出力先")
    args = parser.parse_args()

    if args.repeat < 1 or args.interval < 0:
        parser.error("--repeat は1以上、--interval は0以上にしてください")

    jobs = [
        (model, question, repeat)
        for question in args.questions
        for repeat in range(1, args.repeat + 1)
        for model in args.models
    ]
    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    client = genai.Client(api_key=get_api_key())

    print(f"比較する条件: {len(args.models)}モデル × {len(args.questions)}質問 × {args.repeat}回 = {len(jobs)}リクエスト")
    print(f"リクエスト間隔: {args.interval}秒 / 保存先: {output}")
    try:
        with output.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for index, (model, question, repeat) in enumerate(jobs, 1):
                row = run_one(client, model, question, repeat)
                writer.writerow(row)
                f.flush()  # 途中で制限に達しても測定結果を残す
                rows.append(row)
                print(f"[{index}/{len(jobs)}] {model} / {repeat}回目 / {row['status']} "
                      f"/ {row['elapsed_sec']:.2f}秒 / 合計{row['total_tokens'] or '不明'} tokens")
                if row["status"] != "ok":
                    print(f"  {row['answer']}")
                if row["status"] == "rate_limited":
                    print("429を検出したため、残りの実験を停止します。")
                    break
                if index < len(jobs):
                    time.sleep(args.interval)
    finally:
        client.close()

    print_summary(rows)
    print(f"\n回答本文・各実行の詳細: {output}")


if __name__ == "__main__":
    main()
