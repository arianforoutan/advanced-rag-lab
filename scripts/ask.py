"""Ask the Insurellm agent a question from the command line.

    python scripts/ask.py "What products does Insurellm offer?"
    python scripts/ask.py            # uses a sample question

Set the log level to WARNING to hide the retrieval trace:

    python scripts/ask.py --quiet "..."
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RagPipeline, ask, build_agent  # noqa: E402

DEFAULT_QUESTION = (
    "Which companies have active contracts for the product Homellm, "
    "and what other products does Insurellm offer to them?"
)


def main() -> None:
    args = sys.argv[1:]
    quiet = "--quiet" in args
    args = [arg for arg in args if arg != "--quiet"]

    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    question = " ".join(args).strip() or DEFAULT_QUESTION

    pipeline = RagPipeline.build()
    agent = build_agent(pipeline)
    answer = ask(agent, question)

    print("\n" + "=" * 80)
    print(f"Q: {question}")
    print("-" * 80)
    print(answer)


if __name__ == "__main__":
    main()
