from google.adk.agents import LlmAgent
from .tools import exit_loop

refiner_agent = LlmAgent(
    name = "refiner_agent",
    model = "gemini-2.0-flash",
    description = "피드백을 적용하여 글을 수정하거나, 완료 신호가 감지되면 루프를 종료합니다",
    instruction = """
    당신은 창의적인 글쓰기 비서를 맡고 있습니다. 
    피드백을 바탕으로 이야기 글을 개선하거나, 종료 조건을 판단하세요.

    **현재 이야기 글(Current Story):**
    ```
    {current_story}
    ```
    **피드백 내용(Critique/Suggestions):**
    {criticism}

    **작업 지침(Task):**
    - 피드백이 *정확히* "No major issues found." 와 일치할 경우:
        → 반드시 `exit_loop` 함수를 호출해야 합니다. 아무 텍스트도 출력하지 마세요.
    - 그 외 (피드백이 실제 개선 제안을 포함하는 경우):
        → 해당 제안을 바탕으로 글을 개선한 후, 개선된 글 본문만 출력하세요.

    추가 설명 없이 개선된 이야기 글 또는 루프 종료 함수를 호출하세요.
    """,
    output_key = "current_story",
    include_contents = "none",
    tools = [exit_loop]
)