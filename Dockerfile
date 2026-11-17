FROM mambaorg/micromamba:2.0.5

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml pyproject.toml README.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER mol_design_agents.py run-07.py realtime_bridge.py ./
RUN micromamba create --yes --name moldesign-demo --file environment.yml \
    && micromamba clean --all --yes

USER root
RUN mkdir -p /app/output /app/runinfo \
    && chown -R $MAMBA_USER:$MAMBA_USER /app
USER $MAMBA_USER

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/bin/bash", "-lc", "micromamba run -n moldesign-demo python run-07.py"]
