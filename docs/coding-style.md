# Python coding style

Use multiline docstrings, including short descriptions:

```python
def load_state(path: Path) -> ComputeState:
    """
    Validate the saved recovery manifest.
    """
    return ComputeState.load(path)
```

Add `from __future__ import annotations` and annotate function parameters and
return values. Use ordinary type expressions rather than quoted forward references.
Keep annotation-only imports under `TYPE_CHECKING` when importing them would load
optional inference dependencies or introduce a package cycle. HTTP request models
belong at module scope so FastAPI can resolve their annotations.

Use a `NamedTuple` for small records and Pydantic models for data requiring
validation, serialization, or wider use. Homogeneous sequences and lookup mappings
remain lists, tuples, or dictionaries. Raw JSON dictionaries are appropriate at
cloud SDK and HTTP boundaries; application results should have named fields.
When converting between models, map fields explicitly or provide a named factory
such as `Response.from_outcome(outcome)` that makes the mapping explicit. Avoid
unpacking `model_dump()` into another model. HTTP handlers can return typed models
directly and let FastAPI serialize them.
`Any` is reserved for dynamic SDK payloads, schema fragments, and heterogeneous
resource values. Runtime dependency containers may hold arbitrary live objects;
they are not persistence models. The sealed `Sanitized` dataclass remains immutable
to protect its attestation contract.

Recovery manifests and template reports use the Pydantic models in
`templates/state.py`. They stay in JSON, preserve unknown top-level fields and
replace files atomically. Keep resource identity checks and lifecycle locking when
changing persistence. Consider `pydantic-store` for future persisted dictionaries
or model collections where its storage model fits the use case.

Prefix helpers with an underscore when callers must use a related public operation
that enforces a contract, or when overriding a framework hook. Ordinary reusable
helpers do not need the prefix. Keep comments that explain decisions; repeated
section dividers are a reason to reconsider module boundaries.

Run:

```sh
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run pytest -m 'not external'
```

Ruff checks import order, function annotations in application source, and quoted
forward references. Tests are exempt from annotation linting. Multiline docstrings
remain a review convention: Ruff's one-line docstring rule requires the opposite
layout.
