from fastapi import APIRouter

from app.graph.neo4j_service import Neo4jService
from app.graph.prerequisite_validator import sync_default_prerequisite_graph


router = APIRouter(prefix="/graph", tags=["knowledge-graph"])


@router.get("/status")
def graph_status() -> dict[str, bool | int | str]:
    service = Neo4jService()
    status = service.connectivity_status()
    return {
        **status,
        "skill_nodes": service.count_skills(),
        "prerequisite_edges": service.count_prerequisite_edges(),
    }


@router.post("/sync-prerequisites")
def sync_prerequisites() -> dict[str, bool | int | str]:
    return sync_default_prerequisite_graph()
