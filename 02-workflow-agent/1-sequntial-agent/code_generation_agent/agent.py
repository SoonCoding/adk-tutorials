from google.adk.agents import SequentialAgent
from .sub_agents.writer import code_writer_agent
from .sub_agents.reviewer import code_reviewer_agent
from .sub_agents.refactor import code_refactor_agent

code_pipeline_agent = SequentialAgent(
    name= "code_pipeline_agent",
    description= "코드작성 > 리뷰 > 리팩토링의 순서로 작업을 수행하는 시퀀셜 에이전트입니다",
    sub_agents=[code_writer_agent, code_reviewer_agent, code_refactor_agent]
)

root_agent = code_pipeline_agent