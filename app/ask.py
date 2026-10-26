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
