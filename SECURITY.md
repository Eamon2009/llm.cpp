# Security Policy

Quadtrix.cpp is a local-first transformer learning project. The repository includes native C++ code, Python inference and training code, a FastAPI backend, a React frontend, model checkpoints, and sample datasets. Security reports are welcome, especially when they affect local execution, backend API exposure, dependency safety, or handling of model/data files.

## Supported Versions

Security fixes are considered for:

| Version | Supported |
| --- | --- |
| `master` / active development branch | Yes |
| Latest tagged release | Yes |
| Older releases | Best effort |

If a fix affects both the C++ runtime and the Python or web paths, please call that out in the report so the patch can cover the whole stack.

## Reporting a Vulnerability

Please do not open a public issue for a vulnerability that could put users at risk.

Preferred reporting path:

1. Use GitHub's private vulnerability reporting or security advisory flow for this repository, if available.
2. Include enough detail to reproduce the issue locally.
3. Share the affected component: C++ runtime, Python engine, FastAPI backend, React frontend, packaging, model files, or documentation.

Helpful details include:

- Affected commit, branch, or release.
- Operating system and runtime versions.
- Exact command or request used to reproduce the issue.
- Expected behavior and actual behavior.
- Logs, stack traces, or crash output.
- Whether the issue requires a crafted model, dataset, prompt, HTTP request, or environment variable.

You should receive an acknowledgement as soon as the report is reviewed. Fix timing depends on impact and complexity.

## Security Scope

In scope examples:

- Memory safety bugs in the native C++ runtime.
- Crashes or denial-of-service issues caused by malformed input files, prompts, checkpoints, or HTTP requests.
- Backend API behavior that exposes local files, environment variables, model paths, prompts, sessions, or generated text unexpectedly.
- Unsafe dependency updates or dependency confusion risks in Python, npm, or GitHub Actions.
- Cross-site scripting, service worker, or PWA issues in the frontend.
- Secret leakage through logs, generated artifacts, bundled files, or example configuration.

Out of scope examples:

- Expected model hallucinations or low-quality generated text.
- Prompt injection against a local toy model without a concrete data exposure or code execution path.
- Reports that require already having arbitrary code execution on the user's machine.
- Vulnerabilities in third-party services or packages unless the repository uses them in an unsafe way.

## Local Deployment Notes

Quadtrix.cpp is intended to run locally during development.

- Keep the FastAPI backend bound to `127.0.0.1` unless you intentionally expose it.
- Do not commit real `.env` files, tokens, API keys, private datasets, or private checkpoints.
- Treat downloaded model checkpoints and datasets as untrusted inputs.
- Rebuild native binaries from source when possible instead of trusting unknown executables.
- Review changes to service workers, PWA manifests, backend routing, and CORS settings carefully.

## Dependency Updates

The project uses C++, Python, npm, and GitHub Actions. When updating dependencies:

- Prefer minimal version bumps tied to a clear reason.
- Run the relevant local checks from `contributing.md`.
- Watch for lockfile changes and generated build output.
- Note any security-related update in the pull request summary.
