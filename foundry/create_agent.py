"""Create the employee agent in the Foundry project and test it end to end.

    python foundry/create_agent.py
    python foundry/create_agent.py --ask "How many hours did Jonas Weber work?"

    user question -> agent (gpt-4.1-mini) decides it is an employee-data question
                  -> calls the OpenAPI tool askEmployeeModel
                  -> Azure ML endpoint: the from-scratch employee model
                  -> raw model output comes back, the agent relays it verbatim

The tool is called with the project's managed identity (Entra token for
https://ml.azure.com); no keys anywhere. That identity needs the
AzureML Data Scientist role on the endpoint - grant_endpoint_role() does it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (OpenApiManagedAuthDetails, OpenApiManagedSecurityScheme,
                                    OpenApiTool, RunStepToolCallDetails)
from azure.identity import AzureCliCredential

HERE = Path(__file__).resolve().parent
RG, ENDPOINT, PROJECT = "docintel-ml-rg", "employee-from-scratch", "docintel-finance"
AGENT_NAME, MODEL = "docintel-employee-agent", "gpt-4.1-mini"

INSTRUCTIONS = """You are the front end for a small language model that was trained from scratch on
ten employee documents (timesheets and expense reports). For any question about an
employee, a timesheet, hours, overtime, an expense claim, an employee id (EMP-nnnn) or
a project code (PRJ-nnnn), ALWAYS call the askEmployeeModel tool with the user's question
passed through unchanged, and reply with the tool's `answer` field exactly as returned.
Never rewrite, correct or add to it. If the answer names a different person than the one
asked about, add one sentence: the model only knows its ten documents and that person is
probably not among them. For anything that is not an employee-data question, answer normally."""

AZ = shutil.which("az") or shutil.which("az.cmd") or "az"


def az(*args: str) -> str:
    return subprocess.check_output([AZ, *args], text=True).strip()


def workspace() -> str:
    return az("ml", "workspace", "list", "-g", RG, "--query", "[0].name", "-o", "tsv")


def ai_services_account() -> str:
    return az("cognitiveservices", "account", "list", "-g", RG,
              "--query", "[?kind=='AIServices'].name | [0]", "-o", "tsv")


def project_endpoint() -> str:
    return f"https://{ai_services_account()}.services.ai.azure.com/api/projects/{PROJECT}"


def grant_endpoint_role() -> None:
    """The project identity must be allowed to invoke the endpoint."""
    sub = az("account", "show", "--query", "id", "-o", "tsv")
    url = (f"https://management.azure.com/subscriptions/{sub}/resourceGroups/{RG}/providers/"
           f"Microsoft.CognitiveServices/accounts/{ai_services_account()}/projects/{PROJECT}"
           f"?api-version=2025-04-01-preview")
    principal = az("rest", "--method", "get", "--url", url, "--query", "identity.principalId", "-o", "tsv")
    scope = az("ml", "online-endpoint", "show", "-n", ENDPOINT, "-g", RG, "-w", workspace(),
               "--query", "id", "-o", "tsv")
    existing = az("role", "assignment", "list", "--assignee", principal, "--scope", scope,
                  "--query", "[?roleDefinitionName=='AzureML Data Scientist'] | length(@)", "-o", "tsv")
    if existing == "0":
        az("role", "assignment", "create", "--assignee-object-id", principal,
           "--assignee-principal-type", "ServicePrincipal",
           "--role", "AzureML Data Scientist", "--scope", scope)
        print(f"granted AzureML Data Scientist on {ENDPOINT} to project identity {principal}")
    else:
        print("project identity already has AzureML Data Scientist on the endpoint")


def load_spec() -> dict:
    spec = yaml.safe_load((HERE / "employee-model.openapi.yaml").read_text(encoding="utf-8"))
    uri = az("ml", "online-endpoint", "show", "-n", ENDPOINT, "-g", RG, "-w", workspace(),
             "--query", "scoring_uri", "-o", "tsv")
    spec["servers"] = [{"url": uri.rsplit("/score", 1)[0]}]
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", action="append")
    args = ap.parse_args()

    grant_endpoint_role()
    client = AgentsClient(endpoint=project_endpoint(), credential=AzureCliCredential())

    tool = OpenApiTool(
        name="employee_model",
        description="Sends an employee-data question to the from-scratch employee model.",
        spec=load_spec(),
        auth=OpenApiManagedAuthDetails(
            security_scheme=OpenApiManagedSecurityScheme(audience="https://ml.azure.com")),
    )

    agent = next((a for a in client.list_agents() if a.name == AGENT_NAME), None)
    if agent:
        agent = client.update_agent(agent.id, model=MODEL, instructions=INSTRUCTIONS,
                                    tools=tool.definitions)
        print(f"updated agent {agent.id}")
    else:
        agent = client.create_agent(model=MODEL, name=AGENT_NAME, instructions=INSTRUCTIONS,
                                    tools=tool.definitions)
        print(f"created agent {agent.id}")

    questions = args.ask or ["What are the total hours for EMP-8373?",
                             "Did Farhan Malik work overtime?",
                             "What is the capital of France?"]
    thread = client.threads.create()
    for q in questions:
        client.messages.create(thread_id=thread.id, role="user", content=q)
        run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
        print("=" * 78)
        print(f"Q  {q}")
        if run.status != "completed":
            print(f"   run {run.status}: {run.last_error}")
            continue
        msgs = list(client.messages.list(thread_id=thread.id))
        reply = next(m for m in msgs if m.role == "assistant")
        text = "".join(getattr(c, "text").value for c in reply.content if hasattr(c, "text"))
        called = any(isinstance(s.step_details, RunStepToolCallDetails)
                     for s in client.run_steps.list(thread_id=thread.id, run_id=run.id))
        print(f"A  {text}")
        print(f"   tool called: {'yes' if called else 'NO'}")


if __name__ == "__main__":
    main()
