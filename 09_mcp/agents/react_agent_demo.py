"""실행중인 Weather MCP Tool로 사용될 LangChain 기반 Agent"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from mcp import Client

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import asyncio
import inspect

dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

WEATHER_MCP_URL = "http://127.0.0.1:8000/mcp"
MATH_MCP_URL = "http://127.0.0.1:8001/mcp"

async def _call_mcp_tool(server_url: str, tool_name: str, arguments: dict) -> dict:
    """공식 MCP SDK v2 Client로 실행 중인 HTTP Server의 Tool을 호출한다."""
    async with Client(server_url) as client:
        result = await client.call_tool(tool_name, arguments)

    if result.is_error:
        raise RuntimeError(f"MCP Tool 실행에 실패하였다: {tool_name}")
    return dict(result.structured_content or {})

async def request_weather_without_model(city: str) -> dict:
    """모델 없이 Weather MCP Server의 현재 날씨 Tool을 직접 호출한다."""
    return await _call_mcp_tool(
        WEATHER_MCP_URL,
        "current_weather",
        {"city": city},
    )

def create_math_tool():
    """Math MCP Tool을 호출하는 LangChain Tool wrapper를 만든다."""

    @tool
    async def add_via_mcp(a: float, b: float) -> float:
        """Math MCP Server를 사용해 두 수를 더한다."""
        payload = await _call_mcp_tool(
            MATH_MCP_URL,
            "add",
            {"a": a, "b": b},
        )
        return float(payload["result"])

    return add_via_mcp

def create_weather_tool():
    """Weather MCP Tool을 호출하는 LangChain Tool wrapper를 만든다."""

    @tool
    async def get_current_weather_via_mcp(city: str) -> dict:
        """Weather MCP Server에서 도시의 현재 날씨를 조회한다."""
        return await request_weather_without_model(city)

    return get_current_weather_via_mcp

async def build_agent():
    """Math, Weather MCP wrapper를 사용하는 Agent"""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPEN_API_KEY 누락")

    model = ChatOpenAI(
        model = "gpt-5.6-luna",
        use_responses_api = True
    )

    return create_agent(
        model = model,
        tools = [create_math_tool(), create_weather_tool()],
        system_prompt = (
            "더하기 계산은 add_via_mcp를 사용하고"
            "현재 날씨 조회는 get_current_weather_via_mcp를 사용한다."
            "Tool 결과에 없는 정보는 추측하지 않는다."
        )
    )

# async, await, ainvoke : 순서대로 동작하도록 동기화 -> 안정성 보장
async def ask_weather_agent(question: str) -> dict:
    """Agent가 자연어 날씨 질문에 맞는 MCP Tool을 선택하도록 요청한다."""
    agent = await build_agent()
    return await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )

async def demo_queries(agent) -> None:
    """Weather와 Math MCP Tool을 차례로 호출한다."""
    weather_result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Seoul, KR의 현재 날씨를 날씨 도구로 조회해서 "
                        "기온, 습도와 날씨 설명을 알려줘."
                    ),
                }
            ]
        }
    )
    print(
        "\n[A] === WEATHER ANSWER ===\n",
        weather_result["messages"][-1].content,
    )

    math_result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "3과 7을 수학 도구로 더해줘.",
                }
            ]
        }
    )
    print(
        "\n[B] === MATH ANSWER ===\n",
        math_result["messages"][-1].content,
    )

async def main() -> None:
    """MCP 연결을 확인하고 허용된 경우 원본의 두 Agent 질의를 실행한다."""
    weather_payload = await request_weather_without_model("Seoul")
    print("local MCP result:", weather_payload)

    run_paid_agent = os.getenv("RUN_PAID_AGENT", "false").lower() == "true"
    if not run_paid_agent:
        print("RUN_PAID_AGENT=false: 유료 Agent 호출을 실행하지 않았다.")
        return

    agent = await build_agent()
    await demo_queries(agent)


if __name__ == "__main__":
    asyncio.run(main())

# LLM에게 MCP 호출하는 것을 도구로 등록해
# LLM이 사용자 질문에 맞춰 알아서 MCP를 호출
# (기존 -> 사람이 MCP 이름을 명시해야 함)