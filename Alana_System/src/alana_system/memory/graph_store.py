import sqlite3
import logging
from pathlib import Path
from typing import List, Dict

from alana_system.preprocessing.entity_extractor import KnowledgeGraph

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Persistência local do Knowledge Graph usando SQLite.

    Objetivos:
    - Simplicidade
    - Persistência confiável
    - Base para GraphRAG
    """

    def __init__(self, db_path: str = "data/memory/alana_graph.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ======================================================
    # Infra
    # ======================================================

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Cria tabelas e índices se não existirem."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                # -------------------------
                # Entidades
                # -------------------------
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        type TEXT NOT NULL,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # -------------------------
                # Relações
                # -------------------------
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS relations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        object TEXT NOT NULL,
                        source_doc TEXT,
                        page_number INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(subject, relation, object, source_doc)
                    )
                """)

                # -------------------------
                # Índices (performance)
                # -------------------------
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_timestamp ON relations(timestamp)"
                )

                conn.commit()

                logger.info(f"Grafo SQLite inicializado em: {self.db_path}")

        except sqlite3.Error:
            logger.exception("Erro ao inicializar GraphStore")
            raise

    # ======================================================
    # Escrita
    # ======================================================

    def add_knowledge(
        self,
        graph: KnowledgeGraph,
        source_doc: str,
        page_number: int,
    ) -> None:
        """
        Persiste entidades e relações extraídas do texto.
        """
        if not graph.entities and not graph.relations:
            logger.debug("Grafo vazio recebido — nada para persistir")
            return

        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                # -------------------------
                # Entidades
                # -------------------------
                for entity in graph.entities:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO entities (name, type)
                        VALUES (?, ?)
                        """,
                        (entity.name, entity.type),
                    )

                # -------------------------
                # Relações
                # -------------------------
                for rel in graph.relations:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO relations
                        (subject, relation, object, source_doc, page_number)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            rel.subject,
                            rel.relation,
                            rel.object,
                            source_doc,
                            page_number,
                        ),
                    )

                conn.commit()

                logger.debug(
                    f"Grafo persistido | doc={source_doc} page={page_number} "
                    f"entities={len(graph.entities)} relations={len(graph.relations)}"
                )

        except sqlite3.Error:
            logger.exception("Erro ao persistir conhecimento no GraphStore")

    # ======================================================
    # Leitura
    # ======================================================

    def query_relations(
        self,
        entity_name: str,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Retorna relações conectadas a uma entidade.
        Ideal para expansão de contexto em RAG.
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        subject,
                        relation,
                        object,
                        source_doc,
                        page_number,
                        timestamp
                    FROM relations
                    WHERE subject LIKE ? OR object LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (
                        f"%{entity_name}%",
                        f"%{entity_name}%",
                        limit,
                    ),
                )

                return [dict(row) for row in cursor.fetchall()]

        except sqlite3.Error:
            logger.exception("Erro ao consultar relações do grafo")
            return []

    def count_entities(self) -> int:
        """Retorna o total de entidades conhecidas."""
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM entities")
                return int(cursor.fetchone()[0])
        except sqlite3.Error:
            logger.exception("Erro ao contar entidades")
            return 0
