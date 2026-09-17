"""Ask the deployed from-scratch employee model questions, no document supplied.

    python serving/test_endpoint.py
    python serving/test_endpoint.py --ask "How many hours did Jonas Weber work?"

Authentication is an AAD token from `az login` - the endpoint has no keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request

DEMO = [
    "How many hours did Jonas Weber work?",
    "Did Farhan Malik work overtime?",
    "What is the status of Aisha Rahman's expense report?",
    "What project was Chloe Nguyen on?",
    "How much did Grace Kim claim in expenses?",      # not in the ten -> refusal
]

AZ = shutil.which("az") or shutil.which("az.cmd") or "az"


def az(*args: str) -> str:
    return subprocess.check_output([AZ, *args], text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="employee-from-scratch")
    ap.add_argument("--rg", default="docintel-ml-rg")
    ap.add_argument("--workspace", help="defaults to the only workspace in the resource group")
    ap.add_argument("--ask")
    args = ap.parse_args()

    ws = args.workspace or az("ml", "workspace", "list", "-g", args.rg, "--query", "[0].name", "-o", "tsv")
    uri = az("ml", "online-endpoint", "show", "-n", args.endpoint, "-g", args.rg, "-w", ws,
             "--query", "scoring_uri", "-o", "tsv")
    token = az("account", "get-access-token", "--resource", "https://ml.azure.com",
               "--query", "accessToken", "-o", "tsv")

    for q in ([args.ask] if args.ask else DEMO):
        body = json.dumps({"question": q}).encode()
        req = urllib.request.Request(uri, data=body, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            res = json.loads(r.read())
        print(f"Q  {q}\nA  {res['answer']}   [{res['latency_ms']} ms]\n")


if __name__ == "__main__":
    main()
