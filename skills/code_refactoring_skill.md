---
name: python-refactor
description: >-
  Systematic Python refactoring that turns complex, hard-to-understand code into
  clear, maintainable code while preserving behavior. Includes scope cleanup:
  remove code/files unused outside the user-stated feature scope, and merge
  near-duplicate implementations keeping the most complete one. Applies to OOP,
  procedural, and mixed styles. Use when the user asks for readable, maintainable,
  clean, or refactored code; during reviews; for legacy modernization; or when
  spaghetti, dead code, or duplicate helpers appear.
---

# Python Refactor

Transform complex code into clear, well-documented, maintainable code **without
changing behavior**. Style-agnostic: OOP, procedural, and mixed code are all
valid — choose structure that fits the problem and the surrounding codebase.

Also run a **scope cleanup** pass: drop code/files not used within the
user-stated feature scope, and consolidate functionally similar implementations
into the single most complete one.

## Core principles (priority order)

1. **Clarity over cleverness** — explicit beats clever.
2. **Preserve correctness** — all tests must pass; behavior identical.
3. **One job per unit** — each function, class, or module does one thing well.
4. **Self-documenting structure** — structure says *what*; comments say *why*.
5. **Progressive disclosure** — reveal complexity in layers (helpers, modules).
6. **Match local style** — improve within the existing paradigm unless a style change is explicitly requested or clearly necessary for clarity.
7. **Scope-bounded cleanup** — only remove or merge against the **user-stated** feature scope; never invent scope.
8. **One canonical implementation** — when several units do the same job, keep the most complete and migrate callers.
9. **Reasonable performance** — never sacrifice >2× without explicit approval.

## Key constraints

| Rule | Requirement |
|------|-------------|
| Safety by design | Create new → search all usages → migrate all → verify → only then remove old |
| Static analysis first | `flake8 --select=F821,E0602` (or ruff) before tests after each change |
| Clear boundaries | Make inputs/outputs explicit; avoid hidden mutable globals |
| User-defined scope | Cleanup deletions require an explicit feature-scope statement from the user |
| Preserve in-scope behavior | All behavior inside the stated scope must remain identical |
| No perf regression | Never degrade >2× without approval |
| No API changes | Public APIs unchanged unless requested and documented |
| No over-engineering | Don't add classes, layers, or frameworks just for style |
| No magic | No metaprogramming/framework magic unless necessary |
| Validate continuously | Static analysis + tests after each logical change |

**Any regression = total refactoring failure.** Stop → revert → analyze → fix approach → retry.

## Regression prevention checklist

**Before any session:**

- [ ] Test suite passes 100%
- [ ] Coverage ≥80% on target code (else write tests first)
- [ ] Golden outputs captured for critical edge cases
- [ ] Static analysis baseline saved

**After every micro-change (not only at the end):**

- [ ] `flake8 --select=F821,E999` → 0 errors (or ruff equivalent)
- [ ] `pytest -x` → all pass
- [ ] Behavior unchanged (spot-check one edge case)

## Workflow

### Phase 1 — Analysis

1. Read the full target section for context; note the prevailing style (OOP,
   procedural, mixed) and keep it unless change is justified.
2. List readability issues: nested conditionals, long functions, magic values,
   cryptic names, mixed abstraction levels, tangled state.
3. Assess structure (paradigm-neutral):
   - Are responsibilities clear and separable?
   - Is mutable state explicit (params / return values) or hidden globals?
   - Are dependencies injectable or passable for tests?
   - Do related helpers live together (module or class)?
4. Measure metrics (cyclomatic/cognitive complexity, length, nesting,
   docstring/type coverage).
5. Check test coverage gaps; fill before refactoring.
6. Note cleanup candidates for Phase 2: unused/out-of-scope files, near-duplicate
   helpers/scripts (do not delete yet).
7. Output: prioritized issues by impact and risk.

```python
# ❌ Hard to maintain — tangled responsibilities + hidden state
_cache = {}

def run(user_id):
    global _cache
    if user_id in _cache:
        raw = _cache[user_id]
    else:
        raw = fetch(user_id)          # network
        _cache[user_id] = raw
    if not raw or "@" not in raw["email"]:
        return None
    # ... 80 more lines: validate, transform, persist, format ...

# ✅ Better — same procedural style, clearer units + explicit deps
def get_user(user_id: int, cache: dict, fetch_fn=fetch) -> dict | None:
    if user_id not in cache:
        cache[user_id] = fetch_fn(user_id)
    return cache[user_id]

def is_valid_user(raw: dict) -> bool:
    return bool(raw) and "@" in raw.get("email", "")

def run(user_id: int, cache: dict | None = None, fetch_fn=fetch) -> ...:
    cache = {} if cache is None else cache
    raw = get_user(user_id, cache, fetch_fn)
    if not is_valid_user(raw):
        return None
    ...
```

Classes are fine when they fit (stateful collaborators, protocols, polymorphism).
Plain functions and modules are fine when they fit. Do not convert one into the
other as a default goal.

### Phase 2 — Scope cleanup

Run after analysis, **before** deep structural refactoring. Goal: shrink the
working set to what the user’s feature scope actually needs, then eliminate
duplicate implementations.

#### 2.1 Lock the feature scope (mandatory)

Ask / confirm with the user a concrete scope statement, for example:

```markdown
## Feature scope
- In scope: <features, entrypoints, scripts, configs the user named>
- Out of scope: <explicitly excluded features, experiments, alternate pipelines>
- Keep even if unused in-scope: <tests, public API surfaces, configs user wants retained>
```

**Do not invent scope.** If the user has not named a feature boundary, skip
deletions and only report candidates for confirmation.

#### 2.2 Remove out-of-scope / unused code and files

Treat a symbol or file as a **cleanup candidate** only when **both** are true:

1. It is **not required** by any in-scope entrypoint, call chain, config, or script
   the user named.
2. Evidence of non-use is documented (imports, references, CLI wiring, data paths).

Search beyond Python imports — shells, YAML/JSON configs, `__init__` re-exports,
dynamic `importlib` / string paths, and docs that are the real entrypoint.

```bash
# Example: trace in-scope entrypoints, then find orphans
rg -n "from package.mod import|import package.mod" --glob "*.py"
rg -n "tools/foo\\.py|scripts/.*/train_bar\\.sh" --glob "*.{py,sh,yml,yaml,md}"
```

Classification:

| Label | Meaning | Action |
|-------|---------|--------|
| In-scope used | Reached from stated entrypoints | Keep; refactor later if needed |
| In-scope unused | Inside scope dirs but never referenced by in-scope flows | Candidate to delete (confirm) |
| Out-of-scope only | Only serves features user excluded | Delete after migration checklist |
| Ambiguous | Dynamic import, plugin hook, external caller possible | **Keep** or ask — never delete silently |

Removal uses the same destructive protocol as Phase 3 (checklist → migrate →
verify → remove). Prefer deleting whole unused files over leaving dead stubs.
Update imports/`__all__`/scripts that pointed at removed paths.

#### 2.3 Merge functionally similar code (keep the most complete)

Find near-duplicates: same responsibility, overlapping API, copy-pasted helpers,
parallel scripts/tools that differ only in flags or coverage.

For each cluster:

1. **Inventory** capabilities of each variant (inputs, outputs, edge cases,
   error handling, tests, callers).
2. **Pick the keeper** — the **most complete** implementation:
   - Covers the union of needed behaviors for in-scope callers
   - Best error handling / type hints / tests among equals
   - Prefer the one already used by the primary in-scope entrypoint
3. **Port missing bits** from losers into the keeper **before** switching callers
   (so the keeper becomes a true superset for in-scope needs).
4. **Migrate all callers** to the keeper; then remove the others (files or symbols).

```markdown
## Merge plan: <capability name>
### Candidates
- path/a.py::foo — missing: retries; callers: 2
- path/b.py::foo_v2 — most complete; callers: 5  ← KEEPER
- path/c.py::run_foo — thin wrapper duplicate
### Port into keeper
- [ ] retries from a.py
### Caller migration
- [ ] ...
### Delete after verify
- [ ] path/a.py::foo
- [ ] path/c.py::run_foo
```

**Do not** “average” several half-broken copies into a new fourth implementation
unless no existing variant can absorb the others cleanly — and then still end
with **one** canonical unit.

#### 2.4 Cleanup checklist

- [ ] Feature scope written and acknowledged
- [ ] Out-of-scope / unused candidates listed with evidence
- [ ] Ambiguous items kept or confirmed with user
- [ ] Duplicate clusters scored; keeper chosen; missing features ported
- [ ] All callers migrated; static analysis + tests green
- [ ] Loser symbols/files removed; no broken imports or scripts

Output of this phase: cleanup report (removed paths, merge decisions) feeding
Phase 3 planning.

### Phase 3 — Planning

Classify each change:

- **Non-destructive** (rename, docs, type hints) → low risk
- **Destructive** (remove symbols, reshape APIs, move state, delete files,
  merge duplicates) → high risk; migration plan mandatory

For every destructive change:

```bash
grep -rn "<element_name>" --include="*.py" > migration_plan_<element>.txt
grep -rn "<element_name>\[" --include="*.py" >> migration_plan_<element>.txt
grep -rn "<element_name>\." --include="*.py" >> migration_plan_<element>.txt
```

Document:

```markdown
## Removal Plan: <element_name>
### Total Usages Found: X
### Files Affected: Y
### Detailed Usage List:
- file.py:123 - function_name() - [usage type]
### Migration Strategy:
1. Create replacement structure
2. Migrate usages in order: [...]
3. Verify with static analysis
4. Remove old code
### Risk Level: [High/Medium/Low]
```

If you cannot account for **all** usages, do not proceed with the destructive change.

Also document: risk per change, dependents, test strategy, safest→riskiest order,
expected metric gains, rollback plan.

### Phase 4 — Execution

#### Non-destructive (anytime)

- Rename for clarity
- Extract magic numbers/strings to named constants
- Improve docs and type hints
- Add guard clauses to reduce nesting

#### Destructive — mandatory order

1. **CREATE** new structure + tests; do not remove old code yet
2. **SEARCH** all usages (grep patterns above) → checklist
3. **MIGRATE** one usage at a time; after each: static analysis + tests + commit
4. **VERIFY** checklist 100% and grep finds zero old usages (except new code)
5. **REMOVE** only after verification: comment out → analyze → full tests → delete → commit

Applies to scope-cleanup deletions and duplicate merges as well as local renames.

**Execution rules:**

- Never remove without a verified migration checklist
- Static analysis before tests
- One pattern at a time; atomic commits
- Stop on any error

**Preferred order (safest → riskiest):**

1. Scope cleanup: confirm feature scope
2. Scope cleanup: remove clearly unused / out-of-scope files and symbols
3. Scope cleanup: merge duplicate implementations → one keeper
4. Rename for clarity
5. Extract magic values → named constants
6. Docs + type hints
7. Extract functions/methods (shorten large units)
8. Guard clauses / simplify conditionals
9. Reduce nesting (max 3)
10. Make state and dependencies explicit (params, returns, optional thin wrappers)
11. Group related units (modules and/or classes) only when it reduces confusion
12. Final separation-of-concerns review

### Phase 5 — Validation

```bash
flake8 <file> --select=F821,E0602   # must be zero
flake8 <file> --select=F401
flake8 <file>
pytest
```

Also verify:

- Responsibilities clearer; hidden mutable state reduced or made explicit
- Cleanup: no remaining in-scope references to deleted/merged symbols or files
- In-scope behavior unchanged; out-of-scope removals match the stated scope
- Before/after complexity and quality metrics improved
- Hot-path perf regression check (flag if >10%)
- Flag for human review if: API changed, coverage dropped, architecture shifted,
  flake8 issues increased, or cleanup removed anything ambiguous

## Refactoring patterns

These patterns work in OOP, procedural, and mixed code.

### Complexity reduction

**Guard clauses:**

```python
# Before
def process(data):
    if data:
        if data.is_valid():
            if data.has_permission():
                return data.process()
    return None

# After
def process(data):
    if not data:
        return None
    if not data.is_valid():
        return None
    if not data.has_permission():
        return None
    return data.process()
```

**Extract function/method** — split validation / transform / store into focused units.

**Named boolean / predicate** — replace opaque compound conditions.

**Dictionary dispatch** — replace long if-elif action chains:

```python
HANDLERS = {"create": create, "update": update, "delete": delete}

def process(action, data):
    handler = HANDLERS.get(action)
    if handler is None:
        raise ValueError(f"Unknown: {action}")
    return handler(data)
```

**`match` (3.10+)** — one cognitive-complexity hit for the whole switch.

**Extract to reset nesting** — move nested bodies into helpers so nesting depth drops.

### Naming

| Kind | Convention |
|------|------------|
| Variables | Descriptive; bools `is_`/`has_`/`can_`/`should_`; collections plural |
| Functions | Verb + object; bool queries `is_valid()`, `can_proceed()` |
| Constants | `UPPER_SNAKE`; no magic numbers/strings |
| Classes (when used) | `PascalCase` nouns |

### Documentation

- Docstrings: purpose, Args, Returns, Raises — not implementation narrative
- Module docstring: purpose + key deps
- Inline comments: only non-obvious *why*
- Type hints on all public APIs and complex internals

### Structure (style-agnostic)

- Extract nested logic to helpers (functions or methods)
- Group related behavior (same module, or a class if it owns state/behavior)
- Separate data access, business logic, and presentation when mixed
- Keep consistent abstraction levels inside one function
- Prefer explicit arguments/returns over mutable module globals
- Pass collaborators as parameters (procedural) or inject them (OOP) — both OK
- Use dataclasses/enums when they clarify domain data; not as a default rewrite

**Shared design ideas (apply lightly):** one responsibility per unit; depend on
clear interfaces (functions or protocols), not on hidden concrete wiring; open
for extension when the codebase already uses that pattern.

## Mistakes that break code (never do)

### 1. Incomplete migration

Removing old symbols before every usage is migrated → `NameError` at runtime.

Prevention: full grep checklist → migrate all → static analysis → only then remove.

### 2. Partial pattern application

Fix every instance of a pattern (grep + checklist), not a subset.

### 3. Silent public API breaks

Search all callers; document breaking changes if explicitly approved.

### 4. Trusting tests alone

Tests can miss `NameError`; always run static analysis after each change.

### 5. Style-forced rewrites

Rewriting working procedural code into classes (or the reverse) “because OOP/FP
is better,” without a clarity or testability win — avoid.

### 6. Cleanup without user scope

Deleting “probably unused” code without a stated feature boundary — avoid.
False orphans (plugins, scripts, dynamic imports) cause silent production breaks.

### 7. Merge to the wrong keeper

Picking the shortest or newest duplicate instead of the **most complete**, or
deleting losers before porting their unique behavior into the keeper — avoid.

## Anti-patterns (fix priority)

**Critical first:**

1. Tangled spaghetti — one unit doing fetch + validate + transform + I/O + format
2. Hidden mutable globals / module state that callers cannot see or control
3. God object **or** god module/function with unrelated responsibilities
4. Parallel near-duplicate implementations of the same capability
5. Dead / out-of-scope files kept in the hot path of the stated feature

**High:** nested conditionals >3; functions >30 lines with multiple jobs; magic
values; cryptic names; missing public type hints/docstrings; unclear errors;
mixed abstraction levels

**Medium:** leftover thin wrappers after a merge; primitive obsession; >5
parameters; comments that explain "what"

**Low:** inconsistent naming; redundant comments; unused imports/vars

## Validation tooling

Prefer multi-metric checks; do not rely on one number.

| Metric | Tool | Use |
|--------|------|-----|
| Cognitive complexity | complexipy | Human comprehension |
| Cyclomatic complexity | ruff C901 / radon | Test planning |
| Maintainability | radon | Overall health |
| Lint / bugs / style | ruff (preferred) or flake8 | Quality gate |

```bash
pip install ruff complexipy radon wily

ruff check src/
complexipy src/ --max-complexity-allowed 15
radon mi src/ -s
```

Suggested thresholds:

| Metric | Target | Warn | Error |
|--------|--------|------|-------|
| Cyclomatic / fn | <10 | 15 | 20 |
| Cognitive / fn | <15 | 20 | — |
| Function length | <30 | 50 | — |
| Nesting depth | ≤3 | — | — |
| Public docstring coverage | >80% | — | — |
| Public type-hint coverage | >90% | — | — |

Legacy: raise limits then tighten (e.g. 25 → 20 → 15 → 10). Prefer strict gates
on **changed files only**.

```toml
# pyproject.toml (example)
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "C90", "B", "SIM", "N", "UP", "I"]

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.complexipy]
paths = ["src"]
max-complexity-allowed = 15
exclude = ["tests", "migrations"]
```

Optional flake8 plugin stack (if not on ruff): bugbear, simplify,
cognitive-complexity, pep8-naming, docstrings (+ comprehensions, tryceratops,
annotations as needed).

## Language notes

**Python:** type hints; Google-style docstrings; `pathlib`; specific exceptions;
f-strings / context managers when clear. Dataclasses/enums when they help —
optional, not required.

(Other languages only if the target is not Python: TS explicit types + async/await;
Java interfaces + Optional; Go explicit errors + small interfaces.)

## Output format

```markdown
## Refactoring Summary

**File / scope:** `path/to/file.py` or <feature scope>
**Date:** YYYY-MM-DD
**Risk Level:** [Low/Medium/High]
**Style:** [procedural | OOP | mixed] (preserved / intentionally changed)

### Feature scope
- In scope: …
- Out of scope: …

### Cleanup
- Removed (unused / out-of-scope): `path/a.py`, `mod.unused_helper`, …
- Merged duplicates: `foo_v1` + `foo_legacy` → keeper `foo` (ported: retries)
- Ambiguous kept for confirmation: …

### Changes Made
1. **Extracted function `validate_user_input`** from `process_request`
   - Rationale: ...
   - Risk: Low

### Metrics Improvement
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Avg Cyclomatic Complexity | … | … | … |
| Avg Function Length | … | … | … |
| Max Nesting Depth | … | … | … |
| Docstring Coverage | … | … | … |
| Type Hint Coverage | … | … | … |

### Test Results
- All N tests passing
- Coverage maintained at X%

### Performance Impact
- Hot paths: <2% variance / N/A

### Risk Assessment
**Overall Risk:** …
- Public API changes: yes/no
- Human review needed: yes/no (+ areas)
```

## Edge cases — when not to refactor

- Hot-path / optimized code without profiling + explicit approval
- Code slated for deletion
- Vendored / upstream library code
- Stable legacy that nobody needs to change and risk outweighs benefit

**Limits:** this skill does not change algorithms (e.g. O(n²)→O(n log n)); cannot
invent missing domain meaning; cannot guarantee correctness without tests.

## Success criteria

- [ ] Zero regressions — in-scope tests pass, in-scope behavior unchanged
- [ ] Golden outputs match for critical cases
- [ ] Complexity / docs / type metrics improved
- [ ] No perf regression >10% (or approved)
- [ ] Easier for humans to modify (backed by metrics)
- [ ] No new security issues
- [ ] Atomic, well-documented commits
- [ ] Static analysis improved (issues reduced, not increased)
- [ ] Did not force an unnecessary paradigm rewrite
- [ ] Cleanup followed user-stated scope; ambiguous items not deleted silently
- [ ] Each capability has one canonical implementation; callers migrated

## Related project skills

- `pytorch-coding-style` — fail-fast, tensor contracts, trainer patterns
- `myultralytics-repo-structure` / `myultralytics-scripts-structure` — where code and scripts belong
