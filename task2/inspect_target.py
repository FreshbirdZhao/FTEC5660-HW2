"""Inspect the assignment's target profiles using the supplied MCP connection."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hw2 import load_mcp_tools


async def main() -> None:
    tools = {tool.name: tool for tool in await load_mcp_tools()}
    for name, arguments in [
        ("get_linkedin_profile", {"person_id": 10001}),
        ("get_facebook_profile", {"user_id": 10001}),
    ]:
        print(name)
        print(json.dumps(await tools[name].ainvoke(arguments), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
