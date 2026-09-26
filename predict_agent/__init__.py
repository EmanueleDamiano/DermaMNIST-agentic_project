"""LangGraph prediction agent: local FPViT ensemble + voting + execution log + KB retrieval."""

from predict_agent.agent import as_tool, build_predictor

__all__ = ["as_tool", "build_predictor"]
