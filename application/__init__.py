"""Application layer: workflows that coordinate domain, backend and infrastructure.

The UI calls these services; the services call ``backend`` algorithms and
``infrastructure`` adapters.  No module in this package imports Streamlit, so
the whole application workflow can be tested and reused headlessly.
"""
