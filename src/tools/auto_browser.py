import os
import subprocess
import atexit
import signal
from dotenv import load_dotenv
load_dotenv(verbose=True)

from browser_use import Agent

from src.tools import AsyncTool, ToolResult
from src.tools.browser import Controller
from src.utils import assemble_project_path
from src.registry import TOOL
from src.models import model_manager

@TOOL.register_module(name="auto_browser_use_tool", force=True)
class AutoBrowserUseTool(AsyncTool):
    name = "auto_browser_use_tool"
    description = "A powerful browser automation tool that allows interaction with web pages through various actions. Automatically browse the web and extract information based on a given task."
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task to perform"
            },
        },
        "required": ["task"],
    }
    output_type = "any"

    def __init__(self,
                 model_id: str = "gpt-4.1",
                 max_steps: int = 50,
                 ):
        """
        Args:
            model_id: LangChain-compatible model alias the browser-use Agent
                should drive. Anything without a ``langchain-`` prefix is
                auto-prefixed.
            max_steps: Hard ceiling on the browser-use library's inner
                planning loop per single tool invocation. The default of 50
                matches browser-use's own default and suits production
                research tasks. For smoke / handoff validation runs, drop
                this to 8–15 via ``--cfg-options
                auto_browser_use_tool_config.max_steps=10`` so the agent
                exits the inner loop quickly, producing more outer-loop
                delegations (and therefore more REVIEW / diagnose / skill
                evidence) within the same wall-clock budget.
        """

        super(AutoBrowserUseTool, self).__init__()

        self.model_id = model_id
        self.max_steps = int(max_steps)
        self.http_server_path = assemble_project_path("src/tools/browser/http_server")
        self.http_save_path = assemble_project_path("src/tools/browser/http_server/local")
        os.makedirs(self.http_save_path, exist_ok=True)

        self._init_pdf_server()

    def _init_pdf_server(self):

        server_proc = subprocess.Popen(
            ["python3", "-m", "http.server", "8080"],
            cwd=self.http_server_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=None
        )

        @atexit.register
        def shutdown_server():
            print("Shutting down http.server...")
            try:
                server_proc.send_signal(signal.SIGTERM)
                server_proc.wait(timeout=5)
            except Exception as e:
                print("Force killing server...")
                server_proc.kill()

    async def _browser_task(self, task):
        controller = Controller(http_save_path=self.http_save_path)

        if "langchain" not in self.model_id:
            model_id = f"langchain-{self.model_id}"
        else:
            model_id = self.model_id

        if model_id not in model_manager.registed_models:
            raise ValueError(
                f"Model '{model_id}' is not registered. "
                f"Ensure a LangChain-compatible wrapper is registered under this key."
            )
        model = model_manager.registed_models[model_id]

        browser_agent = Agent(
            task=task,
            llm=model,
            enable_memory=False,
            controller=controller,
            page_extraction_llm=model,
        )

        history = await browser_agent.run(max_steps=self.max_steps)
        contents = history.extracted_content()
        joined = "\n".join(c for c in contents if c)

        if not joined.strip():
            visited = getattr(history, "urls", lambda: [])() or []
            internal_steps = len(history.history) if hasattr(history, "history") else "N"
            raise RuntimeError(
                "auto_browser_use_tool returned no extracted content "
                f"after {internal_steps} internal steps. "
                f"Visited URLs: {visited[:5]}. "
                "Likely causes: page required login/CAPTCHA, JS-heavy content didn't render, "
                "or the target information isn't on the pages visited. "
                "Recommend switching to deep_researcher_agent or refining the task with specific URLs."
            )
        return joined

    async def forward(self, task: str) -> ToolResult:
        """
        Automatically browse the web and extract information based on a given task.

        Args:
            task: The task to perform
        Returns:
            ToolResult with the task result
        """
        try:
            result = await self._browser_task(task)
            return ToolResult(output=result, error=None)
        except Exception as e:
            return ToolResult(output=None, error=str(e))