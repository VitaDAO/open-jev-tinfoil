FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* && useradd --uid 10001 --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin jev
COPY requirements-linux.lock /app/
RUN pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-linux.lock
COPY scripts/download_model.py scripts/pack_weights.py /app/scripts/
RUN python scripts/download_model.py && python scripts/pack_weights.py /opt/model && rm -rf /opt/model/.cache
# Unchanged learned weights are mounted from a separately verified Tinfoil model disk.
# Keep the runtime's original paths so its model identity and behavior stay unchanged.
RUN ln -s /tinfoil/models/jev-learned /opt/learned
COPY vendor /app/vendor
COPY server.py routing.py selector.py learned_selector.py proposal_selector.py trained_proposal_selector.py proposal_binding.py schema_index.py plan_adapter.py query_ir.py direct_selector.py entity_candidates.py temporal_spans.py query_plan.py query_selector.py query_execution.py /app/
COPY learned_parser/*.py /app/learned_parser/
COPY metadata /app/metadata
COPY compat /app/compat
COPY adapters /app/adapters
COPY examples /app/examples
COPY scripts/evaluate_selector_exact.py scripts/run_candidate.py /app/scripts/
ENV MODEL_DIR=/opt/model PYTHONPATH=/app/vendor HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
ENV OPEN_JEV_PARSER=learned LEARNED_PARSER_DIR=/opt/learned/jevparse_r8a,/opt/learned/jevparse_r8b LEARNED_PARSER_MIN_CONFIDENCE=0.9 LEARNED_VERIFIER_DIR=/opt/learned/verifier_v4 LEARNED_VERIFIER_MIN=0.0
USER 10001:10001
EXPOSE 8080
CMD ["python", "/app/scripts/run_candidate.py"]
