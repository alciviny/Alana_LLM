import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Literal, Optional

from ..inference.llm_engine import LLMEngine

logger = logging.getLogger(__name__)

# =========================
# Domain Models (Contratos)
# =========================

EntityType = Literal["Pessoa", "Lugar", "Projeto", "Conceito", "Data", "Organização"]

@dataclass(frozen=True)
class Entity:
    name: str
    type: EntityType

@dataclass(frozen=True)
class Relation:
    subject: str
    relation: str
    object: str

@dataclass
class KnowledgeGraph:
    entities: List[Entity]
    relations: List[Relation]


# =========================
# Entity Extractor
# =========================

class EntityExtractor:
    """
    Converte texto bruto em um Grafo de Conhecimento local
    usando um LLM para extração semântica.
    """

    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def extract_graph(self, text: str) -> KnowledgeGraph:
        """
        Analisa o texto e extrai entidades e relações estruturadas.
        Retorna sempre um KnowledgeGraph válido.
        """
        prompt = self._build_prompt()

        try:
            resposta_bruta = self.llm.generate_answer(
                query=prompt,
                context_text=text,
                max_tokens=1024,
            )

            logger.debug("Resposta bruta do LLM:\n%s", resposta_bruta)

            data = self._safe_json_load(resposta_bruta)
            return self._parse_graph(data)

        except Exception as e:
            logger.exception("Falha crítica na extração de entidades")
            return KnowledgeGraph(entities=[], relations=[])

    # =========================
    # Internals
    # =========================

    def _build_prompt(self) -> str:
        return """
Você é um sistema de extração de conhecimento.

Tarefa:
A partir do TEXTO fornecido, extraia um GRAFO DE CONHECIMENTO.

Regras OBRIGATÓRIAS:
- Responda APENAS com JSON válido.
- NÃO escreva explicações.
- NÃO use markdown.
- NÃO adicione texto antes ou depois do JSON.

Formato exato da resposta:
{
  "entities": [
    {"name": "string", "type": "Pessoa|Lugar|Projeto|Conceito|Data|Organização"}
  ],
  "relations": [
    {"subject": "string", "relation": "string", "object": "string"}
  ]
}

Regras semânticas:
- Normalize nomes (ex: "Alan Turing", não pronomes).
- Use verbos claros nas relações (ex: "criou", "trabalhou_em").
- Não invente entidades que não existam no texto.
"""

    def _safe_json_load(self, raw_text: str) -> Dict[str, Any]:
        """
        Extrai e valida JSON de forma segura a partir da resposta do LLM.
        """
        try:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            json_str = raw_text[start:end]
            return json.loads(json_str)
        except Exception as e:
            logger.error("Erro ao decodificar JSON do LLM")
            logger.debug("Texto recebido:\n%s", raw_text)
            raise ValueError("JSON inválido retornado pelo LLM") from e

    def _parse_graph(self, data: Dict[str, Any]) -> KnowledgeGraph:
        """
        Converte o dicionário cru em objetos de domínio tipados.
        """
        entities: List[Entity] = []
        relations: List[Relation] = []

        for e in data.get("entities", []):
            if "name" in e and "type" in e:
                entities.append(Entity(name=e["name"], type=e["type"]))

        for r in data.get("relations", []):
            if {"subject", "relation", "object"} <= r.keys():
                relations.append(
                    Relation(
                        subject=r["subject"],
                        relation=r["relation"],
                        object=r["object"],
                    )
                )

        return KnowledgeGraph(
            entities=entities,
            relations=relations,
        )
