FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-linux.lock /app/
RUN pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-linux.lock
COPY scripts/download_model.py scripts/pack_weights.py /app/scripts/
RUN python scripts/download_model.py && python scripts/pack_weights.py /opt/model && rm -rf /opt/model/.cache
# Learned request parser weights, staged by scripts/stage_learned_models.py: exactly the pinned files, readable by UID 10001.
COPY learned_models /opt/learned
RUN cd /opt/learned && echo "da75040b34439f604e4c812690ae91f1ae0453ef902c54f51a98c73d70ecb97d  MANIFEST.sha256" | sha256sum --check --strict --quiet && sha256sum --check --strict --quiet MANIFEST.sha256 && test "$(find . ! -type d ! -path ./MANIFEST.sha256 | wc -l)" -eq "$(wc -l < MANIFEST.sha256)" && test -z "$(find . ! -perm -444 -o -type d ! -perm -555)"
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
