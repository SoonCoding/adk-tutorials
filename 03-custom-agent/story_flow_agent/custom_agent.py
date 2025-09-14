from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator
from typing_extensions import override

# --- 사용자 정의 오케스트레이터 에이전트 ---
class StoryFlowAgent(BaseAgent):
    """
    스토리 생성 및 수정 워크플로우를 처리하는 사용자 정의 에이전트입니다.
    이 에이전트는 스토리를 생성하고, 비평 및 수정, 문법 및 톤 체크를 수행하며,
    톤이 부정적일 경우 스토리를 재생성할 수 있습니다.
    """

    # --- Pydantic을 위한 필드 선언 ---
    # 초기화 시 전달된 에이전트들을 클래스 속성으로 선언
    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent

    loop_agent: LoopAgent
    sequential_agent: SequentialAgent

    # 모델 구성이 필요하면 Pydantic 구성을 설정할 수 있습니다. 예: arbitrary_types_allowed
    model_config={"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        story_generator: LlmAgent,
        critic: LlmAgent,
        reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent
    ):
        """
        StoryFlowAgent를 초기화합니다.

        Args:
            name: 에이전트의 이름.
            story_generator: 초기 스토리를 생성하는 LlmAgent.
            critic: 스토리를 비평하는 LlmAgent.
            reviser: 비평을 바탕으로 스토리를 수정하는 LlmAgent.
            grammar_check: 문법을 검사하는 LlmAgent.
            tone_check: 톤을 분석하는 LlmAgent.
        """
        # 내부 에이전트를 생성한 후 super().__init__ 호출
        loop_agent = LoopAgent(
            name = "loop_agent",
            sub_agents = [critic, reviser],
            max_iterations=2
        )

        sequential_agent = SequentialAgent(
            name = "sequential_agent",
            sub_agents = [grammar_check, tone_check]
        )

        # 서브 에이전트 목록 정의
        sub_agents_list = [story_generator, loop_agent, sequential_agent]

        # Pydantic은 클래스 어노테이션을 기반으로 에이전트를 초기화합니다.
        super().__init__(
            name=name,
            story_generator=story_generator,
            critic=critic,
            reviser=reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,
            loop_agent=loop_agent,
            sequential_agent=sequential_agent,
            sub_agents=sub_agents_list  # 서브 에이전트 목록을 직접 전달
        )

    @override
    async def _run_async_impl(
        self, ctx :InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        스토리 워크플로우를 위한 사용자 정의 오케스트레이션 로직을 구현합니다.
        Pydantic에 의해 할당된 인스턴스 속성(예: self.story_generator)을 사용합니다.
        """

        # 1. 스토리 초안 생성
        async for event in self.story_generator.run_async(ctx):
            yield event  # 초기 스토리 생성 실패 시 처리 중지

        # 2. 스토리 생성 여부 확인 (스토리가 생성되지 않으면 워크플로우를 중단)
        if 'current_story' not in ctx.session.state or not ctx.session.state['current_story']:
            return

        # 3. 스토리 비평 & 수정 반복 작업
        async for event in self.loop_agent.run_async(ctx):
            yield event

        # 4. 스토리 문법 검사 & 톤 분석 순차 작업
        async for event in self.sequential_agent.run_async(ctx):
            yield event

        # 5. 스토리 톤 분석 결과가 부정적(negative)일 경우 스토리를 재생성
        tone_check_result = ctx.session.state.get("tone_check_result")
        # 톤이 부정적일 경우 스토리 재생성
        if tone_check_result == "negative":
            async for event in self.story_generator.run_async(ctx):
                yield event
        else:
            # 톤이 부정적이지 않으면 현재 스토리를 유지
            pass

        # 워크플로우가 종료