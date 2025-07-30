from google.adk.agents import ParallelAgent, SequentialAgent
from .sub_agents.renewable_energy import renewable_energy_agent
from .sub_agents.ev_techonology import ev_technology_agent
from .sub_agents.carbon_capture import carbon_capture_agent
from .sub_agents.synthesizer import synthesis_agent

parallel_research_agent = ParallelAgent(
    name = "parallel_research_agent",
    description = "여러 연구 에이전트를 병렬로 실행하여 정보를 수집하는 에이전트",
    sub_agents = [renewable_energy_agent, ev_technology_agent, carbon_capture_agent]
)

sequential_pipeline_agent = SequentialAgent(
    name = "sequential_pipeline_agent",
    description = "병렬 연구 실행 후 결과를 통합하는 에이전트",
    sub_agents = [parallel_research_agent, synthesis_agent]
)

root_agent = sequential_pipeline_agent

