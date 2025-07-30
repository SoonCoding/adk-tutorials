from google.adk.agents import LlmAgent
from google.adk.tools import google_search

carbon_capture_agent = LlmAgent(
    name = "carbon_capture_agent",
    model = "gemini-2.0-flash",
    description = "탄소 포집 방법을 연구하는 에이전트",
    instruction = """
    당신은 기후 솔루션 분야 AI 연구 보조입니다.
    '탄소 포집 방법'의 현재 상태를 Google 검색 도구를 사용해 조사하세요.
    핵심 발견 내용을 간결하게 1~2문장으로 요약하고,
    **요약만** 출력하세요.
    """,
    tools = [google_search],
    output_key = "carbon_capture_result"
)