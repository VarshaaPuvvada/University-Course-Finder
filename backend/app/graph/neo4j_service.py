import os

from app.utils.env import load_backend_env


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

    def sync_prerequisites(self, prerequisites: dict[str, list[str]]) -> None:
        driver = self._driver()
        if driver is None:
            return
        with driver.session() as session:
            for skill, required_skills in prerequisites.items():
                for required in required_skills:
                    session.run(
                        """
                        MERGE (required:Skill {name: $required})
                        MERGE (skill:Skill {name: $skill})
                        MERGE (required)-[:PREREQUISITE_FOR]->(skill)
                        """,
                        required=required,
                        skill=skill,
                    )
        driver.close()

    def expand_prerequisites(self, skills: set[str]) -> set[str]:
        driver = self._driver()
        if driver is None:
            return set()
        expanded: set[str] = set()
        with driver.session() as session:
            for skill in skills:
                rows = session.run(
                    """
                    MATCH (required:Skill)-[:PREREQUISITE_FOR*1..3]->(skill:Skill {name: $skill})
                    RETURN DISTINCT required.name AS name
                    """,
                    skill=skill,
                )
                expanded.update(row["name"] for row in rows if row["name"])
        driver.close()
        return expanded

