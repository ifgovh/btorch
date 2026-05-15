# AGENT.md

Btorch is a brain-inspired Torch library for neuromorphic research. Follow the patterns in the codebase: stateful Modules, careful tensor shape/dtype handling, and practical usability over over-engineering.

## Repo focus

- Primary package: `btorch/` (models, connectome tools, analysis, visualization).
- Docs live in `docs/en/docs/`; examples in `examples/`.
- Avoid adding new heavy dependencies unless explicitly requested.
- Connectome tooling lives under `btorch/connectome/`; keep data transforms explicit and reversible.
- Neuron models live under `btorch/models/neurons/`; follow existing parameter/state patterns.

## Coding guidelines

- Match existing module structure and naming.
- Use `jaxtyping` for tensor shapes where it clarifies intent.
- Use modern Python type annotations (`|`, `list`, `dict`); avoid
  `typing.Optional`/`typing.Union` unless absolutely required.
- Be explicit about device, dtype, and batch/time dimensions.
- Preserve torch.compile compatibility and ONNX friendliness where applicable.
- Prefer buffers/register_memory for stateful tensors; validate shapes/dtypes.
- Keep code and comments within the formatter line length (88 chars); wrap long docstrings and comments.
- Use English for code, comments, and docstrings; conversation can be in other languages.
- In tests, include thorough explanatory comments so they are understandable and can serve as examples.

## Documentation format

Use **Google style** docstrings (preferred) with this structure:

```python
"""One-line summary (imperative, no period).

Extended description. Explain what, not how. Use math notation for equations:

.. math::
    \tau \frac{dv}{dt} = -(v - v_{reset}) + R \cdot I(t)

Args:
    param1: Description with type inferred from annotation.
    param2: Multi-line descriptions
        should indent.

Returns:
    Description of return value.

Raises:
    ValueError: When input is invalid.

Examples:
    Basic usage:

    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output

Notes:
    Additional implementation details.

References:
    [1] Author, Title, Journal, Year.
"""
```

Key patterns:

- Document `__init__` parameters in class docstring, not `__init__` method
- Include mathematical model equations in `.. math::` blocks
- Include cross-references with `:func:` and `:class:`
- For every function you generate, document the motivation for the function,
  every argument, and the return value or output behavior. If a function has
  side effects, document those explicitly.

## Tests and checks

- Lint: `ruff check .`
- Tests: `pytest tests` or targeted subfolders
- Docs: `python scripts/docs.py command=build-all` (builds all languages into `site/`)

## Documentation

- Update `docs/en/docs/` for user-facing changes.
- Never touch the auto generated `docs/en/docs/api`.
- Update `README.md` for installation or workflow changes.

## Contribution workflow

- Use pre-commit formatting before PRs: `pre-commit install --install-hooks`.
- Keep changes modular; avoid mixing refactors with feature work.
- Every time you make code or documentation changes, create a git commit for
  that change set unless the user explicitly tells you not to.
- After every such commit, update a Markdown progress report that records the
  commit ID and summarizes the purpose, key changes, and verification status.
- Do not frequently stop to ask for confirmation. Make reasonable assumptions
  and finish as much of the plan as possible before asking, unless the choice
  is risky, destructive, or blocked by missing information.

## Design principles

- Usability over performance; simple over clever; customizability over abstraction.
- Keep network models cohesive (single file/folder principle).

## Experimental trials and reproducibility

Whenever I run an experimental comparison, benchmark, or parameter-sweep, I
**must** save a complete, self-contained trial record so the result can be
reproduced later.

### Trial folder location

Base path: `/data/guozhang/RESULTS/dendrite/`

Each trial gets a **randomly named sub-folder** (e.g. a UUID or timestamp hash):

```
/data/guozhang/RESULTS/dendrite/
  ├── trial_a7f3e2d9/
  ├── trial_8b12c4f0/
  └── ...
```

If `/data/guozhang/RESULTS/dendrite/` does not exist, I should create it.
If I lack write permissions, I should warn the user and fall back to a
local `trials/` directory inside the repo root.

### What to save inside every trial folder

1. **`REPRODUCE.md`** – Exact reproduction instructions:
   - The exact command(s) run
   - Python / conda environment name
   - Random seeds used
   - Hardware info (CPU/GPU) if relevant

2. **`git_commit.txt`** – Output of `git rev-parse HEAD`

3. **`git_diff.patch`** – Output of `git diff HEAD` (or `git diff` against the
   last committed state). This captures any uncommitted changes I made during
   the trial.

4. **`config.json`** or **`config.yaml`** – All hyper-parameters, model
   settings, and CLI arguments used for the run.

5. **Result artifacts** – Copies of the actual outputs (figures, CSVs, model
   checkpoints, reports) generated by the script.

6. **`logs/`** – Any stdout / stderr logs from the run.

### Notes and progress reports (local folder)

My working notes and the Markdown progress report mentioned in the
**Contribution workflow** section can be kept in a **local folder** inside the
repo (e.g. `notes/` or `progress/`). These do not need to be copied to the
shared `/data/guozhang/RESULTS/dendrite/` drive unless the user explicitly asks
for them to be archived there.

### How to create a trial folder

I should create the folder **before** starting the run, then populate it
incrementally. At minimum, after the run finishes I must run:

```bash
TRIAL_DIR="/data/guozhang/RESULTS/dendrite/trial_$(uuidgen | cut -c1-8)"
mkdir -p "$TRIAL_DIR"
git rev-parse HEAD > "$TRIAL_DIR/git_commit.txt"
git diff > "$TRIAL_DIR/git_diff.patch"
# copy config, results, etc.
```

### Rule of thumb

If I spend more than a few minutes running an experiment, I save it. No
exceptions. Reproducibility is more important than disk space.
