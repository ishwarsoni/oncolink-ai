"""Infrastructure layer: external services, configuration and logging.

Modules are imported explicitly (for example
``from infrastructure.llm import call_llm``) so that importing the package does
not eagerly load ChromaDB, PyTorch or the OpenAI client.
"""
