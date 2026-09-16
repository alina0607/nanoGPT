# Notebooks

Exploration lives here; the implementation does not.

A notebook is the right tool for inspecting tensor shapes, plotting a loss curve,
or poking at an intermediate value while working out what a component should do.
It is the wrong tool for code that needs to be imported, tested, reviewed in a
diff, or trusted six months from now.

## Workflow

1. Explore in a notebook until the mechanism is clear.
2. Move the working version into a module at the repository root.
3. Cover it with a test under `tests/`.
4. Commit the module, the test, and the stripped notebook together.

## Stripping output before committing

Notebook output is stored inline as JSON, which bloats the repository and makes
diffs unreadable. Clear it first:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

To automate this, install [nbstripout](https://github.com/kynan/nbstripout) and
register it as a git filter:

```bash
pip install nbstripout
nbstripout --install
```
