from google.adk.agents import LoopAgent, SequentialAgent

from .sub_agents.writer import initial_writer_agent
from .sub_agents.critic import critic_agent
from .sub_agents.refiner import refiner_agent

refinement_loop = LoopAgent(
    name = "수정반복루프",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=5
)

root_agent = SequentialAgent(
    name="반복적글쓰기파이프라인",
    description="초기 글을 작성한 후, 비판과 함께 반복적으로 수정하여 완성도를 높입니다",
    sub_agents=[initial_writer_agent, refinement_loop]
)