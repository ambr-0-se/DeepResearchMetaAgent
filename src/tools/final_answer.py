from src.tools import AsyncTool, ToolResult
from src.registry import TOOL

@TOOL.register_module(name="final_answer_tool", force=True)
class FinalAnswerTool(AsyncTool):
    name = "final_answer_tool"
    description = (
        "Provides the final answer to the task. Call this ONLY on its own, and "
        "ONLY AFTER you have received and read observations from prior tool calls. "
        "NEVER emit this in the same turn as any other tool call — the answer "
        "must reflect real observations, not a prediction of what tools will return."
    )
    parameters = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "The final answer to the problem. Must be grounded in observations "
                    "already returned by prior tool calls; must not be a predicted/placeholder value."
                ),
            },
        },
        "required": ["answer"],
    }
    output_type = "any"

    async def forward(self, answer: str) -> ToolResult:
        result = ToolResult(
            output=answer,
            error=None,
        )
        return result