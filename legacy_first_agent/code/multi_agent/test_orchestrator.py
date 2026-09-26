graph.add_node("augmentation", augmentation_agent)
graph.add_node("training", training_agent)
graph.add_node("prediction", prediction_agent)
graph.add_node("explainability", explainability_agent)
graph.add_node("evaluation", evaluation_agent)

graph.add_edge(START, "augmentation")
graph.add_edge("augmentation", "training")
graph.add_edge("training", "prediction")
graph.add_edge("prediction", "explainability")
graph.add_edge("explainability", "evaluation")


## langgraph permette un'orchestrazione deterministca, utile per riprodurre gli esperimenti 