class RagPromptService:

    @staticmethod
    def build_prompt(query: str, context: str) -> str:
        return f"""
당신은 제조 장애 분석 지원 AI입니다.

아래 제공된 과거 장애 이력을 근거로만 답변하세요.
제공된 정보에 없는 내용을 임의로 추측하지 마세요.
근거가 부족하면 "현재 제공된 장애 이력만으로는 판단하기 어렵습니다."라고 답변하세요.

답변은 다음 형식으로 작성하세요.

1. 예상 원인
2. 우선 확인할 항목
3. 권장 조치
4. 참고한 장애 이력

[과거 장애 이력]

{context}

[사용자 질문]

{query}
""".strip()
