from google.adk.agents import LlmAgent

code_writer_agent = LlmAgent(
    name= "code_writer_agent",
    model= "gemini-2.0-flash",
    description= "사용자의 요청에 따라 초기 파이썬 코드를 작성합니다",
    instruction= """
    당신은 Python 코드 생성기입니다.  
    사용자의 요청 *내용만을 기반으로* 요구사항을 충족하는 Python 코드를 작성하세요.  
    결과는 *오직* 전체 Python 코드 블록만을 출력하며, 세 개의 백틱(```python ... ```)으로 감싸야 합니다.  
    코드 블록 전후에는 그 외의 어떤 텍스트도 추가하지 마세요.
    """,
    output_key="generated_code"
)
