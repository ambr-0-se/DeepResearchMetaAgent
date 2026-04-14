import copy
import json
from typing import Optional

from src.models import MessageRole, Model, ChatMessage
from src.metric.arc_scorer import extract_grid_from_text
from src.logger import logger


async def prepare_arc_response(original_task: str, inner_messages, reformulation_model: Model) -> Optional[str]:
    """Extract the final ARC grid answer from the agent conversation.

    Returns a JSON string of the 2D grid, or ``None`` on extraction failure.
    """
    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an answer-extraction assistant for ARC-AGI tasks. "
                        "Earlier the following ARC-AGI task was posed:\n\n"
                        f"{original_task}\n\n"
                        "A team then worked to solve it. "
                        "Read the transcript below:"
                    ),
                }
            ],
        }
    ]

    try:
        for message in inner_messages:
            if not message.get("content"):
                continue
            message = copy.deepcopy(message)
            message["role"] = MessageRole.USER
            messages.append(message)
    except Exception:
        messages += [{"role": MessageRole.ASSISTANT, "content": str(inner_messages)}]

    messages.append(
        {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Read the above conversation and extract the FINAL ANSWER grid "
                        "for the ARC-AGI test input.\n\n"
                        "Output ONLY a valid JSON 2D array (list of lists of integers 0-9).\n"
                        "Example: [[0,1,2],[3,4,5]]\n\n"
                        "Do not include any other text, explanation, or formatting. "
                        "Just the raw JSON array."
                    ),
                }
            ],
        }
    )

    messages = [ChatMessage.from_dict(msg) for msg in messages]

    response = await reformulation_model(messages)
    response_text = response.content

    grid = extract_grid_from_text(response_text)
    if grid is not None:
        result = json.dumps(grid)
        logger.info(f"> ARC reformulated answer: {result}")
        return result

    logger.warning(f"> ARC reformulation failed to extract grid from: {response_text[:300]}")
    return None
