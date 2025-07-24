from google.adk.agents import LlmAgent

STATE_INITIAL_TOPIC = "30년 뒤 미래"
initial_writer_agent = LlmAgent(
    name = "initial_writer_agent",
    model = "gemini-2.0-flash",
    description = "정해진 주제를 바탕으로 초안 글을 생성합니다",
    instruction = f"""
    당신은 창의적인 이야기 작성을 돕는 AI 비서입니다.
    아래 주제를 기반으로 짧은 이야기의 *첫 번째 초안*을 작성하세요. (2-5문장 정도)

    주어진 주제에 *집중하여*, 독자의 흥미를 끌 수 있도록 등장인물, 배경, 사건 중 하나 이상의 구체적인 요소를 포함하세요.
    주제: {STATE_INITIAL_TOPIC}

    이야기 본문만 출력하세요. 설명이나 서론은 작성하지 마세요.
    """,
    output_key = "current_story",
    include_contents = "none"
)