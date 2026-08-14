from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.llm_service import LlmService


def main():
    prompt = """
다음 내용을 한 문장으로 요약하세요.

Motor temperature exceeded threshold.
Cooling fan malfunction was identified.
The cooling fan was replaced and temperature returned to normal.
""".strip()

    print("=== LLM Test ===")
    print()

    answer = LlmService.generate(prompt)

    print(answer)


if __name__ == "__main__":
    main()
