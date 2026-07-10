# Contributing to Quadtrix.cpp

Thanks for helping improve Quadtrix.cpp. This project is a transformer learning lab with several execution paths: native C++ training and inference, PyTorch experiments, a FastAPI backend, and a React + TypeScript chat UI. Contributions are easiest to review when they keep those paths clear and testable.

## Good First Contributions

Useful contributions include:

- Fixing correctness bugs in the C++ transformer implementation.
- Improving training, inference, checkpoint loading, or export scripts.
- Adding focused documentation for setup, model files, datasets, or run commands.
- Tightening CI, dependency versions, packaging, or release steps.

For larger model architecture changes, open an issue first so the design can be discussed before a big patch lands.

If you cannot run a relevant check, mention that in the pull request and explain why.

## Pull Request Guidelines

- Keep changes focused on one problem or feature.
- Use the existing style of the file you are editing.
- Avoid committing generated artifacts unless the project already expects them.
- Do not commit `.env` files, secrets, private datasets, or personal checkpoints.
- Update `README.md`, `run.md`, or related docs when commands or behavior change.
- Include screenshots or short notes for UI changes.
- Mention any change that affects model files, ports, CORS, service workers, or packaging.

The pull request template asks for:

- Summary and user-facing impact.
- C++ build status.
- Documentation or screenshot updates when needed.

## Coding Notes

For C++ changes:

- Prefer clear, debuggable code over clever abstractions.
- Keep the educational value of the implementation visible.
- Be careful with tensor shapes, bounds, and ownership.
- Add comments only where the math or control flow is not obvious.

For Python changes:

- Keep backend behavior explicit and local-development friendly.
- Avoid broad exception swallowing around model loading or inference.
- Treat model paths, datasets, and request payloads as untrusted inputs.


## Documentation Style

Use concrete commands and paths. llm.cpp has multiple runtime paths, so say exactly which path a command belongs to: C++, PyTorch.

When documenting training results, include the hardware, dataset, iteration count, elapsed time, and validation metric so results can be compared fairly.
