"""Push the current INSTRUCTIONS to the migrated (versioned) copy of the agent.

    python foundry/publish_version.py

Why this exists: create_agent.py talks to the classic Assistants API. When
you click "Save as new agent" in the portal, Foundry copies the agent into
the versioned agent API, and from then on the two are independent - editing
the classic one does not touch the copy the playground and Copilot use.
This script publishes a new version of that copy with the instructions from
create_agent.py, keeping its model and tools as they are.
"""

from __future__ import annotations

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.identity import AzureCliCredential

from create_agent import AGENT_NAME, INSTRUCTIONS, project_endpoint


def main() -> None:
    client = AIProjectClient(endpoint=project_endpoint(), credential=AzureCliCredential())
    current = next((a for a in client.agents.list() if a.name == AGENT_NAME), None)
    if current is None:
        raise SystemExit(f"{AGENT_NAME} has not been migrated yet - click 'Save as new agent' in the portal first")
    definition = current.versions["latest"]["definition"]
    tools = [t for t in definition["tools"] if t.get("type") != "web_search"]   # the portal adds this; drop it
    new = client.agents.create_version(
        AGENT_NAME, definition=PromptAgentDefinition(model=definition["model"], instructions=INSTRUCTIONS, tools=tools))
    print(f"published {new.id}")


if __name__ == "__main__":
    main()
