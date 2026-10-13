# Copilot Instructions

- Keep `app.py` in the project root. The Render entry command must remain `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`.
- Place application logic in `resume_screener/`; keep the root app focused on Streamlit UI and orchestration.
- Preserve existing dependency versions unless a task explicitly authorizes upgrades. Put development-only dependencies in `requirements-dev.txt`.
- Use environment-backed configuration with safe defaults. Never commit secrets or place API keys in module-global mutable state.
- Log unexpected exceptions with the project logger and show concise, actionable UI errors. Do not silently disguise analysis failures as valid scores.
- Validate uploaded files and user input before processing; escape dynamic content rendered as HTML.
- Add or update focused pytest coverage for behavior changes. Keep tests independent of network access and external services.
- Keep changes scoped to the requested behavior and do not commit unless explicitly asked.
