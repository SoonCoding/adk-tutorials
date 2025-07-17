from google.adk.agents import LlmAgent

root_agent = LlmAgent(
    name="welcome_agent",
    model="gemini-2.0-flash",
    description="인사하는 에이전트",
    instruction="""
    당신은 사용자를 환영하고 인사하는 에이전트 입니다.
    사용자의 이름을 물어보고, 그 이름으로 환영 인사를 해주세요.
    """
)