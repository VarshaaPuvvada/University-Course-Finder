import os
import logging
from contextlib import contextmanager
from typing import Iterator

from app.utils.env import load_backend_env

logging.getLogger("neo4j").setLevel(logging.CRITICAL)


class Neo4jService:
    def __init__(self) -> None:
        load_backend_env()
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

    @property
    def enabled(self) -> bool:
        return bool(self.uri and self.username and self.password)

    def _driver(self):
        if not self.enabled:
            return None
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return None
        return GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    @contextmanager
    def _session(self) -> Iterator:
        driver = self._driver()
        if driver is None:
            yield None
            return
        try:
            with driver.session() as session:
                yield session
        finally:
            driver.close()

    def connectivity_status(self) -> dict[str, bool | str]:
        if not self.enabled:
            return {
                "enabled": False,
                "connected": False,
                "message": "Neo4j credentials are not configured.",
            }

        driver = self._driver()
        if driver is None:
            return {
                "enabled": True,
                "connected": False,
                "message": "Neo4j driver is not installed.",
            }
        try:
            driver.verify_connectivity()
        except Exception as exc:
            return {
                "enabled": True,
                "connected": False,
                "message": str(exc).splitlines()[0],
            }
        finally:
            driver.close()

        return {
            "enabled": True,
            "connected": True,
            "message": "Neo4j connection is available.",
        }

    def sync_prerequisites(self, prerequisites: dict[str, list[str]]) -> int:
        synced = 0
        try:
            with self._session() as session:
                if session is None:
                    return synced
                for skill, required_skills in prerequisites.items():
                    for required in required_skills:
                        session.run(
                            """
                            MERGE (required:Skill {key: $required_key})
                              ON CREATE SET required.name = $required_name
                            MERGE (skill:Skill {key: $skill_key})
                              ON CREATE SET skill.name = $skill_name
                            MERGE (required)-[:PREREQUISITE_FOR]->(skill)
                            """,
                            required_key=required.lower(),
                            required_name=required,
                            skill_key=skill.lower(),
                            skill_name=skill,
                        )
                        synced += 1
        except Exception:
            return 0
        return synced

    def expand_prerequisites(self, skills: set[str]) -> set[str]:
        expanded: set[str] = set()
        try:
            with self._session() as session:
                if session is None:
                    return expanded
                for skill in skills:
                    rows = session.run(
                        """
                        MATCH (required:Skill)-[:PREREQUISITE_FOR*1..3]->(skill:Skill {key: $skill})
                        RETURN DISTINCT coalesce(required.name, required.key) AS name
                        """,
                        skill=skill.lower(),
                    )
                    expanded.update(row["name"] for row in rows if row["name"])
        except Exception:
            return set()
        return expanded

    def count_skills(self) -> int:
        try:
            with self._session() as session:
                if session is None:
                    return 0
                row = session.run("MATCH (skill:Skill) RETURN count(skill) AS total").single()
                return int(row["total"]) if row else 0
        except Exception:
            return 0

    def count_prerequisite_edges(self) -> int:
        try:
            with self._session() as session:
                if session is None:
                    return 0
                row = session.run(
                    "MATCH (:Skill)-[edge:PREREQUISITE_FOR]->(:Skill) RETURN count(edge) AS total"
                ).single()
                return int(row["total"]) if row else 0
        except Exception:
            return 0
