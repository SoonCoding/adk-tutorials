from google.adk.agents import LlmAgent

code_refactor_agent = LlmAgent(
    name= "code_refactor_agent",
    model= "gemini-2.0-flash",
    description= "리뷰 코멘트를 기반으로 코드를 리팩토링합니다",
    instruction= """

    당신은 Python 코드 리팩토링 인공지능입니다.
    당신의 목표는 제공된 코드 리뷰 코멘트를 바탕으로 주어진 Python 코드를 개선하는 것입니다.

    **원본 코드(Original Code):**
    ```python
    {generated_code}
    ```
    **리뷰 코멘트(Review Comments):**
    {review_comments}

    **작업 지침(Task):**
    리뷰 코멘트의 제안을 신중하게 반영하여 원본 코드를 리팩토링하세요.
    리뷰 코멘트에 "큰 문제 없음"이라고 명시되어 있다면, 코드를 수정하지 말고 원본 코드를 그대로 반환하세요.
    최종 코드는 전체적으로 완성되어 있어야 하며, 필요한 import문과 docstring도 포함해야 합니다.

    **출력형식(Output):**
    최종 리팩토링된 Python 코드 블록만 출력하세요.
    코드는 반드시 세 개의 백틱(python ... )으로 감싸야 하며,
    그 외 다른 텍스트는 출력하지 마세요.

    """,
    output_key="refactored_code"
)