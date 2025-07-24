from google.adk.agents import LlmAgent

critic_agent = LlmAgent(
    name = "critic_agent",
    model = "gemini-2.0-flash",
    description = "초안글을 검토하고 필요한 경우 개선 피드백을 제공하거나 완료되었음을 신호합니다",
    instruction = """
    당신은 창의적인 글을 비평하는 AI 비평가입니다. 이야기 초안(일반적으로 2~6문장)을 검토하고, 균형 잡힌 피드백을 제공하세요.

    **검토할 초안(Document to Review):**
    ```
    {current_story}
    ```

    **작업 지침(Task):**
    초안의 주제 적합성, 명확성, 흥미 유발 요소 등을 기준으로 평가하세요..

    - 만약 1~2개의 *명확하고 실행 가능한* 개선점을 제시할 수 있다면 
      (예: "도입 문장이 더 강력해야 합니다", "등장인물의 목표를 명확히 하세요"):
    	→ 간결하게 피드백을 작성하세요. 피드백 텍스트만 출력하세요.

    - 반대로, 초안이 주제를 충실히 다루고 있고, 눈에 띄는 오류나 누락된 점이 없다면:
    	→ 정확히 이 문구만 출력하세요: "No major issues found."

    추가 설명 없이 피드백 또는 해당 완료 문구만 출력하세요.
    """,
    output_key = "criticism",
    include_contents = "none"
)