from app.graph.prerequisite_validator import sync_default_prerequisite_graph


if __name__ == "__main__":
    result = sync_default_prerequisite_graph()
    for key, value in result.items():
        print(f"{key}: {value}")
