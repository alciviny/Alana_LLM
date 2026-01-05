import logging
import threading
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
            # Fallback para uso máximo da GPU
            n_gpu_layers = -1

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
        self._lock = threading.Lock()

    def generate_answer(self, query: str = None, context_text: str = None, messages: list = None) -> str:
        try:
            with self._lock:
                # Se recebermos uma lista de mensagens (usado pelo EntityExtractor)
                if messages:
                    output = self.llm.create_chat_completion(
                        messages=messages,
                        temperature=0.1, # Baixa temperatura para extração de dados
                        max_tokens=1024
                    )
                else:
                    # Fallback para o modo de busca comum
                    prompt = f"Contexto: {context_text}\n\nPergunta: {query}\nResposta:"
                    output = self.llm.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024
                    )

            return output["choices"][0]["message"]["content"].strip()

        except RuntimeError as e:
            # Captura especificamente o erro de estouro de contexto ou erro de decode
            if "llama_decode returned -1" in str(e):
                logger.error("⚠️ Erro de Contexto: O bloco de texto é muito complexo ou longo para o LLM. Pulando este chunk...")
            else:
                logger.error(f"❌ Erro de Runtime no LLM: {e}")
            return "" # Retorna vazio para o extrator ignorar e seguir em frente

        except Exception as e:
            logger.error(f"❌ Erro inesperado no LLM Engine: {e}")
            return ""
