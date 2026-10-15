# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

I use a LangChain tool-calling agent with `deepseek-v4-flash`, temperature 0,
and the supplied SocialGraph MCP tools. The agent first resolves identity using
name, location, industry, and independent employment/education anchors, with
broader searches when a potentially false claim would otherwise exclude the
candidate. It retrieves full LinkedIn profiles and uses Facebook when useful
for identity corroboration. It then audits every claimed name, city, job,
education entry, and skill, allowing equivalent wording and omitted information
but rejecting even one inconsistent year or inflated qualification. The final
structured audit is converted into 0.95 for a complete match, 0.05 for a
discrepancy, or 0.50 for unresolved evidence; these are decision scores, not
calibrated probabilities. A positive score requires coverage of all five field
categories and a recorded LinkedIn profile tool response. CV text is treated as
untrusted data to resist embedded instructions. Up to three CVs run concurrently;
each has two attempts of at most 110 seconds, bounded model requests and graph
steps, and an isolated failure fallback. The audit is recovered by scanning the
final message for a balanced JSON object instead of parsing the whole string, so
the prose summary and Markdown code fence that the model usually prepends no
longer discard an otherwise correct audit. The provided runner is unchanged.
Public-test results: **7/7**. `CV_1`, `CV_2`, `CV_3` and `CV_6` (valid) score
0.95, and `CV_4`, `CV_5` and `CV_7` (discrepant) score at or below 0.05, so every
prediction matches its label. Three consecutive runs on `public_test` each
returned 7/7.

### Task 2

The one-page `task2/adversarial_cv.pdf` attacks the boundary between PDF rendering
and text extraction. The visible education line claims **MSc in Finance,
graduated 2016, HKUST**, upgrading Kelly Tsang's actual **BSc** to a master's
degree. This is a real false degree claim, not just different wording. Other
claims follow the supplied target CV. On 15 October 2026, direct MCP lookups of
LinkedIn person 10001 and Facebook user 10001 confirmed the BSc, employment,
skills, city, and the Facebook alias "Kel Tsang"; Facebook describes the
education as "Bachelor's Degree". The degree line is rendered as an image;
an invisible PDF text layer at the same position says **BSc in Finance**. Thus a
person reading the page sees the embellished degree, while a text-only verifier
can receive the true degree and find no contradiction with the MCP profile.
The attack uses no server modifications or external data sources. Its source
is provided in `task2/build_adversarial_cv.py` for reproducibility.

This exploits representation mismatch rather than relying on a model to obey a
malicious instruction. Prompt-injection defenses cannot recover a false claim
that the loader never supplies to the model. Since the five course verifiers
use the same text loader, the mismatch may transfer across different prompts
and agent designs. Rendering the page and applying OCR, or comparing visual and
text layers, could expose it. Task 1 receives only extracted text and the
assignment forbids changing the PDF loader, so that defense lies outside the
permitted Task 1 interface. Success against my own agent and success against
the five undisclosed agents must be distinguished; only the graders can measure
the latter. Self-test: `python3 hw2.py --cv-folder task2` scores
`adversarial_cv.pdf` at **0.95** and the true `target_cv.pdf` at 0.95, so the
attack succeeds against my own Task 1 agent.

Reproduce the PDF and inspect the course target profiles:

```bash
python task2/build_adversarial_cv.py
python task2/inspect_target.py
python hw2.py --cv-folder task2
```

### Task 3: Reflection on AI paradoxes

**Automation and responsibility.** A verifier can reduce the time needed to
screen CVs, but a score can make an uncertain judgment look objective. In this
assignment, an unresolved identity receives 0.50 and is counted as discrepant
by the evaluation threshold. In a real hiring process, "not verified" should
remain separate from "false": missing evidence is not evidence of dishonesty.
A responsible deployment would offer human review, access to the disputed
claim, and a correction process instead of automatically rejecting a person.

**Capability and vulnerability.** The language flexibility that lets an agent
recognize "Bachelor of Science" as BSc also makes it susceptible to persuasive
instructions mixed into documents. More reasoning does not automatically solve
the problem: in Task 2, a text-based model can reason correctly about every token
it receives and still approve a visually false document. Reliability belongs
to the whole pipeline, including document extraction, identity resolution,
tool responses, and decision rules, rather than to the language model alone.

**Verification and trust.** The assignment defines the MCP profiles as ground
truth, making tool-based verification meaningful. Outside the exercise, two
matching profiles might simply repeat the same false claim. Agreement is not
necessarily independent corroboration. A verifier therefore needs to explain
what it established: consistency with specified sources, rather than absolute
truth about a candidate. More sources are helpful only if their provenance,
independence, and relevance are understood.

**Optimization and understanding.** A system optimized for a small public test
can learn shortcuts that fail on new names, locations, or document formats.
Likewise, a successful adversarial PDF demonstrates a weakness in the evaluated
pipeline, not general superiority over the model. Evaluation should include
unseen identities, isolated field changes, extraction failures, and ambiguous
cases. The useful outcome of building and attacking the verifier is a clearer
account of its assumptions and failure modes, not merely a higher score.

### Reproduction notes

Use Python 3.10 or newer (the local environment uses Python 3.12); the macOS
system Python 3.9 does not satisfy the LangChain requirements. Install
`requirements.txt` in a virtual environment and supply `DEEPSEEK_API_KEY` through
the environment or an untracked `.env` file. Run `python hw2.py --cv-folder
public_test` for Task 1 and `python hw2.py --cv-folder task2` for the attack test.
Each command overwrites `results.csv`, so preserve it before running the other
folder. The offline checks cover structured-audit handling, discrepancy
precedence, missing evidence, the concurrency limit, and per-CV failure
isolation; these checks do not substitute for live MCP/model evaluation.
