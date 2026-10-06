# Train a new model from scratch on Azure ML

> Read **[Azure-Document-Ingestion](https://github.com/Auxin-io/Azure-Document-Ingestion#readme)** first. It covers prerequisites. It has produced the data, and the shared foundation exists.

In this project we trains a **new language model from random weights** on the ten employee documents
(timesheets and expense reports), serves it from an Azure ML managed online
endpoint, and puts a Foundry agent in front of it. The model answers
employee questions from its own weights.

```
Blob (employee JSONL)  ->  Azure ML job: causal Transformer from scratch (T4, 77 s)  ->  Model registry
                                                                                             |
User -> Foundry agent (gpt-4.1-mini) -> OpenAPI tool -> Managed online endpoint (CPU) <------+
```

Training data comes from the
[Azure-Document-Ingestion](https://github.com/Auxin-io/Azure-Document-Ingestion).

---

## Workflow diagram

The diagram below shows the workflow of the project. 

<img width="4183" height="1377" alt="AI Project#1 - Doc Intel AWS v2 - PreTraining-Workflow" src="https://github.com/user-attachments/assets/8fb7143a-6b8a-4bad-beb4-b87b107d93a3" />

Builds the word vocabulary from the rows and trains the 4-layer transformer from random weights (3.2M
parameters, 4000 steps, ~77 s on the T4), the checkpoint is registered and served on a CPU
`Standard_DS1_v2`, and the Foundry agent reaches it through the `askEmployeeModel` OpenAPI
tool with the project identity's AAD token.

## Azure services used

| Service | What it does in this project |
|---|---|
| **Blob Storage** (ingestion account) | holds the employee closed-book JSONL under `curated/datasets/closed_book_employee/` |
| **Document Intelligence** | OCR of the employee PDFs (ingestion repo) |
| **AML datastore `ingest_curated`** | lets the workspace read that container without keys |
| **AML data assets** `employee-closed-book-{train,validation,test}` | versioned pointers to the three JSONL files |
| **AML compute cluster `gpu-t4`** | runs `train.py` for 77 s; no pretrained weights are downloaded |
| **AML model registry** | versions the checkpoint (`employee-from-scratch-model`) with its vocabulary inside |
| **AML managed online endpoint** (`employee-from-scratch`, `Standard_DS1_v2`) | serves the 3.2M-parameter model on one CPU core; AAD token auth |
| **Container Registry** | builds the torch-only training and serving images |
| **AI Services account + `gpt-4.1-mini`** | the agent's reasoning model |
| **Foundry project `<project>`** | hosts `employee-agent` and its OpenAPI tool `askEmployeeModel` |
| **Managed identity + Entra ID RBAC** | the project identity gets AzureML Data Scientist on the new endpoint (the script grants it) |
| **Azure Bot Service** (optional, created by Publish) | exposes the agent in Microsoft 365 Copilot |

---

## Prerequisites

- Azure CLI 2.89+ with the ML extension; Terraform >= 1.9; Python 3.11+
- `az login` into a subscription where you are **Owner** (Terraform and the
  steps below assign roles)
- **Azure ML GPU quota** for `Standard NCASv3_T4 Family` in your region. This
  is separate from the Virtual Machines quota — check and request it before
  Step 1:

```bash
az login
az extension add -n ml
```

Portal → **Quotas → Machine Learning → your region → Standard NCASv3_T4
Family**: if the limit is 0, request 12 before continuing (approved within
the hour in our case).

---

## Step 1 — infrastructure

```bash
cd terraform
terraform init
terraform apply
terraform output
cd ..
```

Creates, in `<prefix>-pretrain-rg`: an ML workspace with its storage account,
Key Vault, App Insights and Log Analytics; a Container Registry; one training
cluster at min 0 / max 1; an AI Services account with a `gpt-4.1-mini`
deployment; a Foundry project; a credential-less datastore on the ingestion
container; and the role assignments that make all of it work without keys.

Two things to do after apply. First, attach the registry to the workspace -
Terraform cannot, because setting `container_registry_id` forces the workspace
to be replaced on every later apply:

```bash
eval "$(terraform -chdir=terraform output -raw attach_registry_command)"
```

Second, load the resource names the scripts need. No script hardcodes them:

```bash
eval "$(terraform -chdir=terraform output -raw agent_env)"
```

---

## Step 2 — data

Register the three Blob files as data assets. They are read through the
credential-less datastore:

```bash
cd data
az ml data create -f closed_book_train.yml      -g <ml-rg> -w <workspace>
az ml data create -f closed_book_validation.yml -g <ml-rg> -w <workspace>
az ml data create -f closed_book_test.yml       -g <ml-rg> -w <workspace>
cd ..
```

---

## Step 3 — train

```bash
cd training
JOB=$(az ml job create -f job.yml -g <ml-rg> -w <workspace> --query name -o tsv | tr -d '\r')
echo "$JOB"
```

| Setting | Value | Why |
|---|---|---|
| initialisation | random | the point of the exercise — no pretrained weights anywhere |
| tokenizer | word-level, built from the training rows | ids, dates and amounts stay whole tokens (`EMP-8373`, `2026-05-04`, `42.2`) |
| architecture | 4 pre-norm decoder blocks, d=256, 4 heads, tied output head | 3.2 M parameters |
| sequence | `<bos> question <sep> answer <eos>`, context 96 | rows are ~30 tokens |
| loss | answer tokens + `<eos>` only | learns question → answer, not question text |
| steps / batch | 4000 / 64 | 77 s on the T4 |
| augmentation | random lower-casing, punctuation, greeting prefixes | so it does not key on exact wording |

Watch it:

```bash
az ml job show -n $JOB -g <ml-rg> -w <workspace> --query status -o tsv
```
The step counter is in
Studio → job → *Outputs + logs → user_logs/std_log.txt*.

---

## Step 4 — register and serve

```bash
az ml model create -g <ml-rg> -w <workspace> \
  --name employee-from-scratch-model --type custom_model \
  --path "azureml://jobs/$JOB/outputs/model"

cd ../serving
az ml online-endpoint create   -f endpoint.yml   -g <ml-rg> -w <workspace>
az ml online-deployment create -f deployment.yml -g <ml-rg> -w <workspace> --all-traffic
```

`endpoint.yml` sets `auth_mode: aad_token`. `deployment.yml` runs
`score.py` on a **`Standard_DS1_v2`** instance, a 3 M-parameter
model needs no GPU and answers in 130–300 ms. The scorer walks the mounted
model folder for `employee_model.pt`, which carries its own vocabulary and
architecture config.

Test:

```bash
python serving/test_endpoint.py --ask "How many hours did Jonas Weber work?"
```

```
Q  How many hours did Jonas Weber work?
A  Jonas Weber (EMP-8373) logged 42.2 hours for the week ending 2026-05-04.
```

---

## Step 5 — the Foundry agent

```bash
cd ../foundry
python3 -m venv .venv-agents
source .venv-agents/bin/activate
pip install -r requirements.txt
.venv-agents/bin/python create_agent.py
```

The script creates (or updates)
`employee-agent` on `gpt-4.1-mini` with the endpoint as an OpenAPI
tool authenticated by managed identity, then asks three questions:

```
Q  What are the total hours for EMP-8373?
A  Jonas Weber (EMP-8373) logged 42.2 hours for the week ending 2026-05-04.
   tool called: yes
Q  Did Farhan Malik work overtime?
A  Farhan Malik recorded 0.3 hours of overtime for the week ending 2026-08-18.
   tool called: yes
Q  What is the capital of France?
A  The capital of France is Paris.
   tool called: NO
```

**Portal:** https://ai.azure.com → New Foundry → project `<project>` →
Agents → `employee-agent` → Save as new agent → Playground.

---

## Step 6 — publish to Microsoft 365 Copilot

In the migrated agent click **Publish → Teams and Microsoft 365**, fill in the
descriptions, keep the generated bot name, and finish. This creates an Azure
Bot Service (free F0) and a service principal.

---

## Test questions

The ten documents: timesheets for Jonas Weber, Hugo Lindqvist, Farhan Malik,
Chloe Nguyen, Isabel Moreno; expense reports for Noah Fischer, Keiko Tanaka,
Ben Carter, Aisha Rahman, Elena Petrova.

| Ask | Expect |
|---|---|
| How many hours did Jonas Weber work? | EMP-8373, 42.2 hours, week ending 2026-05-04 |
| Did Farhan Malik work overtime? | 0.3 hours, week ending 2026-08-18 |
| What project was Chloe Nguyen on? | PRJ-3307, approved by S. Brooks |
| What is the status of Aisha Rahman's expense report? | EXP-70486, Reimbursed |
| How much did Noah Fischer claim in expenses? | $1,417.60 on EXP-87838 |
| How many hours did Isabel Moreno work? | EMP-9850, 35.7 hours |
| What is the capital of France? | answered by gpt-4.1-mini, no tool call |

---

## Cost and teardown

| Component | Cost |
|---|---|
| Training | a few cents per run (`gpu-t4` scales to zero) |
| Endpoint on DS1_v2 | **~USD 0.06/hour while deployed** |
| gpt-4.1-mini | per token |

```bash
az ml online-endpoint delete -n employee-from-scratch -g <ml-rg> -w <workspace> -y
cd terraform && terraform destroy  
```

---

## Files

```
data/
  closed_book_{train,validation,test}.yml   data assets on the ingestion Blob datastore
training/
  job.yml            command job on gpu-t4, torch only
  train.py           tokenizer, model, training loop, exact-match evaluation
  environment.yml
serving/
  endpoint.yml, deployment.yml   AAD-only endpoint on a DS1_v2
  score.py                       same tokenizer + model class; greedy decoding
  environment.yml, test_endpoint.py
foundry/
  create_agent.py                agent + OpenAPI tool + role grant + test
  publish_version.py             pushes new INSTRUCTIONS to the migrated (versioned) agent
  employee-model.openapi.yaml    the tool definition; servers[] filled in at run time
  requirements.txt
```
