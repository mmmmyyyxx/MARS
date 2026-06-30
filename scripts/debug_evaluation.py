from __future__ import annotations

import argparse

from mars_core.evaluator import (
    canonical_answer,
    extract_prediction,
    is_correct_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug answer parsing.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--answer-format", required=True)
    args = parser.parse_args()
    print(f"parsed={extract_prediction(args.raw, args.answer_format)}")
    print(f"gold={canonical_answer(args.gold, args.answer_format)}")
    print(f"correct={is_correct_prediction(args.raw, args.gold, args.answer_format)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
