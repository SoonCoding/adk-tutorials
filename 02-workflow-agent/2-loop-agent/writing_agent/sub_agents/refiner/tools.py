from typing import Dict, Any
from google.adk.tools.tool_context import ToolContext

def exit_loop(tool_context: ToolContext) -> Dict[str, Any]:
    """
    루프 종료 도구입니다. 글이 더 이상 개선이 필요 없다고 판단되면 호출됩니다.

    Args:
        tool_context: 도구 실행 시의 컨텍스트

    Returns:
        빈 딕셔너리 (도구는 JSON 직렬화 가능한 값을 반환해야 함)
    """
    
    print("\n----------- 루프 종료 트리거됨 -----------")
    print("비평이 완료되어 루프를 종료합니다.")
    print("----------------------------------------\n")

    tool_context.actions.escalate = True
    return {}
