from google.adk.agents import LlmAgent
from google.adk.tools import google_search

renewable_energy_agent = LlmAgent(
    name = "renewable_energy_agent",
    model = "gemini-2.0-flash",
    description = "재생 가능 에너지 소스 연구 에이전트",
    instruction = """
    당신은 재생 가능 에너지 분야의 AI 연구 보조입니다.
    '재생 에너지 소스'의 최신 발전 동향을 Google 검색 도구를 사용해 조사하세요.
    핵심 발견 내용을 간결하게 1~2문장으로 요약하고,
    **요약만** 출력하세요.
    """,
    tools = [google_search],
    output_key = "renewable_energy_result"
)