from google.adk.agents import LlmAgent
from google.adk.tools import google_search

ev_technology_agent = LlmAgent(
    name = "ev_technology_agent",
    model = "gemini-2.0-flash",
    description = "전기차 기술을 연구하는 에이전트",
    instruction = """
    당신은 교통 분야 AI 연구 보조입니다.
    '전기차 기술'의 최근 발전사항을 Google 검색 도구를 사용해 조사하세요.
    핵심 발견 내용을 간결하게 1~2문장으로 요약하고,
    **요약만** 출력하세요.
    """,
    tools = [google_search],
    output_key = "ev_technology_result"
)