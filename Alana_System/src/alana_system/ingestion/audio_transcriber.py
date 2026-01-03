import logging
from pathlib import Path
from typing import List
import torch
import whisper

from .text_extractor import PageText

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    Transcritor de áudio local usando OpenAI Whisper.
    
    Responsabilidades:
    - Carregar modelo Whisper (base, small, medium, large)
    - Processar áudio e gerar texto bruto
    - Adaptar saída para formato compatível com o pipeline (PageText)
    """

    def __init__(self, model_size: str = "base", device: str = None):
        """
        Args:
            model_size: Tamanho do modelo ('tiny', 'base', 'small', 'medium', 'large')
            device: 'cuda' ou 'cpu'. Se None, detecta automaticamente.
        """
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"Carregando modelo Whisper '{model_size}' em {self.device}...")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
        except Exception as e:
            logger.critical(f"Falha ao carregar Whisper: {e}")
            raise

    def transcribe(self, audio_path: Path) -> List[PageText]:
        """
        Transcreve o áudio e retorna como uma única 'página' de texto.
        
        Nota: Futuramente poderia quebrar em várias 'Pages' baseadas em timestamp
        se o áudio for muito longo (ex: 1 hora).
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")

        logger.info(f"Iniciando transcrição: {audio_path.name}")
        
        try:
            # O Whisper faz todo o trabalho pesado aqui
            # fp16=False evita erros em algumas CPUs
            result = self.model.transcribe(
                str(audio_path), 
                fp16=(self.device == "cuda")
            )
            
            raw_text = result["text"].strip()
            
            # Encapsula num PageText para o resto do sistema aceitar
            # Usamos page_number=1 pois tratamos o arquivo como documento único
            page_text = PageText(
                page_number=1, 
                text=raw_text,
                char_count=len(raw_text)
            )

            logger.info(f"Transcrição concluída | {len(raw_text)} caracteres gerados")
            return [page_text]

        except Exception as exc:
            logger.exception(f"Erro na transcrição de {audio_path.name}")
            raise RuntimeError(f"Falha no Whisper para {audio_path.name}") from exc