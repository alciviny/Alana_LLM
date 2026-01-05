import logging
from typing import Optional
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Engine de LLM local usando llama.cpp

    Responsabilidades:
    - Carregar modelo local (CPU ou GPU)
    - Aplicar prompt seguro e determinístico
    - Gerar respostas baseadas EXCLUSIVAMENTE no contexto
    """

    def __init__(
        self,
        model_path: str,
        context_window: int = 4096,
        n_gpu_layers: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            model_path: Caminho do arquivo .gguf
            context_window: Janela total de contexto (prompt + resposta)
            n_gpu_layers:
                - None  -> auto detecta
                - 0     -> CPU
                - -1    -> tenta usar tudo da GPU
            seed: Seed fixa para respostas determinísticas
        """

        if n_gpu_layers is None:
            # Fallback seguro (CPU)
            n_gpu_layers = 0

        logger.info("🔄 Inicializando LLM local")
        logger.info(f"📦 Modelo: {model_path}")
        logger.info(f"🧠 Context Window: {context_window}")
        logger.info(f"🎮 GPU Layers: {n_gpu_layers}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_window,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )

    def generate_answer(
        self,
        query: str,
        context_text: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        Gera resposta baseada em contexto usando o formato de chat do modelo.
        """

        if not query.strip():
            raise ValueError("Query vazia")

        if not context_text.strip():
            logger.warning("⚠️ Contexto vazio fornecido ao LLM")
        
        system_message = """Você é a Alana, minha assistente pessoal inteligente.

REGRAS:
- Use APENAS o contexto fornecido (minhas notas, áudios e documentos).
- Seja direta, amigável e organize as informações de forma útil.
- Sempre cite a fonte e a página/arquivo. Exemplo: (Fonte: diario.md).
- Se a informação não estiver registrada, diga: "Não encontrei nada sobre isso nas minhas notas."
"""
        
        human_message = f"""CONTEXTO:
{context_text}

PERGUNTA:
{query}
"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message},
        ]

        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=[
                    "<|eot_id|>",
                    "<|end_of_text|>",
                ],
            )

            return output["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.exception("❌ Erro ao gerar resposta do LLM")
            raise RuntimeError("Falha na geração da resposta") from e
