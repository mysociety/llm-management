# Commit plan

This records the implemented commit grouping and the local QuestionSlice history
cleanup. Both fixups were autosquashed, and the five feature commits below were
created in order. Each intermediate snapshot passed its available local tests.
The remote branch has not been updated. Commit IDs in the fixup section refer
to the history before rewriting.

A backup of that history remains at `backup/commit-plan-20261008`, and the original
working changes are retained in the `commit-plan-20261008 working tree backup`
stash. The final local suite passed 118 tests; lint, formatting and bootstrap
wheel packaging checks also passed. Type checking identified a missing assertion
for an existing-template lookup; that assertion was folded into commit 3. Tests
ran with the isolated dependency environment used for live verification. The live
results below came from the earlier
cloud run; committing these changes did not create any new cloud resources.

## Reassign the latest commit

`49dcc4b` (`update question slice`) removes source-unit membership and orphan
indices from the public QuestionSlice response. It updates the implementation,
schemas, tests and documentation together. This is a refinement of the existing
QuestionSlice feature, separate from the new System One work.

Recommend replacing it with two fixups:

| Changes from `49dcc4b` | Fixup target |
| --- | --- |
| Root `README.md`, `foi/question_slice.py`, `foi/schemas.py`, and the three changed test files | `34ea496` — Replace FOI structure agents with QuestionSlice and fine-tuned Granite |
| `src/llm_management/foi/readme.MD` | `91595b6` — add information request readme |

Exact fixup commit messages:

```text
fixup! Replace FOI structure agents with QuestionSlice and fine-tuned Granite
```

```text
fixup! add information request readme
```

Use `git commit --fixup=34ea496` and `git commit --fixup=91595b6`, respectively,
after staging each group. Autosquashing removes these temporary messages and
retains the original target commit messages.

The split matters: `foi/readme.MD` was introduced by `91595b6`, so moving its
changes into `34ea496` would edit a file before it exists. Keep the intervening
deployment-group commit (`2edf8ec`) separate.

Before rewriting, save the working changes, including untracked files, and make
a backup branch. Extract the latest commit's changes relative to its parent,
split them by the paths above, create `fixup!` commits against the two targets,
and autosquash from `34ea496^`. Restore the working changes afterward and check
that the final source tree matches the pre-rewrite tree.

The local branch is one commit ahead of `origin/span-approach`; both fixup targets
are already in the remote branch's history. Autosquashing would therefore rewrite
published history. Coordinate that rewrite before updating the remote, and use
`--force-with-lease` if it is agreed. Creating the new feature commits does not
depend on doing this rewrite.

## New commits, in order

### 1. Update Pydantic AI for native System One support

Commit message:

```text
Update Pydantic AI for native System One support
```

- `pyproject.toml`: upgrade `pydantic-ai-slim` and add `httpx2`.
- `poetry.lock`: corresponding dependency resolution.

Leave bootstrap packaging changes for commit 3. Keep the lockfile content hash
consistent with the manifest at each step by regenerating it as necessary.
Run the existing non-external tests to check compatibility with the major
Pydantic AI upgrade.

### 2. Add System One proxy and typed immigration classification

Commit message:

```text
Add System One proxy and typed immigration classification
```

- `src/llm_management/systemone.py`: request forwarding and transport errors.
- `agents/immigration_detection.py`: typed decision agent and field descriptions.
- `server.py`: System One proxy, native immigration endpoint, their imports and
  error handling, and removal of caller credentials from upstream requests.
- `models.py`: default managed backend and OpenAI protocol fields.
- `server.py`: protocol check for OpenAI agents.
- `tests/test_systemone.py`: authentication, forwarding, typed outputs and failures.
- `README.md`: endpoint entries, request examples and API behavior.

Keep the existing lifecycle and config types at this stage. The endpoint tests
mock the upstream connection, so they do not require a Compute VM or a new
deployment entry. Defer README paragraphs about Compute provisioning and recipe
selection until commit 4.

Validate with `poetry run pytest -m 'not external' tests/test_systemone.py`, then
the existing authentication and agent tests.

### 3. Add named Exoscale template recipes and preparation CLI

Commit message:

```text
Add named Exoscale template recipes and preparation CLI
```

- All files in `src/llm_management/templates/`: validated recipes, name resolution,
  Compute helpers, bootstrap, preparation, cleanup and generic HTTP/native probes.
- `conf/exoscale_templates.toml`: the pinned Clef Flash recipe and general challenges.
- `__main__.py`: register the `templates` subcommands; defer Compute logs dispatch.
- `pyproject.toml`: package `templates/bootstrap.sh` in both wheel and source
  distribution, including the consistent table form for the existing LICENSE entry.
- `poetry.lock`: refresh its manifest hash if needed.
- `tests/test_templates.py` and `tests/test_template_compute.py`.
- `EXOSCALE_TEMPLATES.md`: container source, recipe format, preparation, template
  testing, cleanup and storage behavior.
- `README.md`: template CLI overview and link to the guide.

Use the external `mysociety/systemone-container` repository as the container
source. Template challenges should remain independent of immigration detection.
Defer the guide's main-server deployment and full lifecycle verification sections
until commits 4 and 5.

Validate the two template test modules, the CLI help, and a built wheel's inclusion
of `templates/bootstrap.sh`. These checks do not create cloud resources.

### 4. Add prepared Compute deployments to the shared lifecycle

Commit message:

```text
Add prepared Exoscale Compute deployments to the shared lifecycle
```

- `compute_deployments.py`: durable state, owned resources, provisioning, health
  checks, SSH tunnels, reconnect, logs, probes and deletion.
- `models.py`: Compute config, deployment union, template references and dispatch.
- `settings.py` and `.gitignore`: persistent state location and ignored state files.
- `cache.py`: backend tracking and active-request accounting.
- `server.py`: shared lifecycle dispatch, request leases, startup cancellation,
  idle/shutdown teardown, explicit pause, readiness and lifecycle error responses.
- `foi/backends.py`: shared deployment type annotations.
- `__main__.py`: Compute logs dispatch.
- `conf/exoscale.toml`: `clef` deployment referencing the `clef_flash` recipe.
- `tests/test_compute_deployments.py`: lifecycle and server regression checks.
- `README.md` and `EXOSCALE_TEMPLATES.md`: deployment config, ownership, persistent
  keys, one-manager constraint and Compute teardown behavior.

Keep the safety checks and cleanup tests with this feature. They ensure existing
VMs are reused, incomplete starts remain discoverable, and active requests prevent
teardown. This commit depends on both the template helpers and System One routes.

Validate with `poetry run pytest -m 'not external' tests/test_compute_deployments.py`,
then run the complete non-external suite.

### 5. Add live lifecycle verification and document observed results

Commit message:

```text
Test the live Compute lifecycle and document verified template results
```

- `tests/test_compute_external.py`: explicitly selected live test with an isolated
  role, HTTP/native calls, tunnel recovery, pause and resource cleanup.
- `EXOSCALE_TEMPLATES.md`: verified template name/UUID, timings, cleanup results and
  the repeatable external-test command.
- `README.md`: short summary linking to the live verification section.
- `COMMIT_PLAN.md`: this handover plan, if it is useful to retain in the repository.

Record the already completed live verification: 118 local tests passed; the live
lifecycle test passed; an independent audit found no remaining Compute VMs.
The two private templates were retained. The timings are observations rather than
a startup guarantee. No temporary JSON reports or experiment directories need to
be committed.

Running the live test again creates billed cloud resources. Use the isolated-role
command in the guide when a repeat is needed; the default test run excludes it.

## Staging and final checks

Several groups share `server.py`, `models.py`, `__main__.py`, `pyproject.toml` and
`README.md`. Stage by hunk, splitting hunks where necessary; do not stage those
files wholesale into the first group. New files can be staged by complete path.
Use an isolated checkout or temporarily set aside the remaining changes to verify
each proposed commit without later uncommitted code masking a missing dependency.

For each group, inspect `git diff --cached` and run `git diff --cached --check`.
After the complete stack, run `script/test`, the relevant lint/type checks and a
package build. Confirm that documentation links resolve and the external test
remains opt-in. Removing the untracked experiment folders creates no Git deletion
commit; their useful behavior is now covered by the package, tests and root guide.
