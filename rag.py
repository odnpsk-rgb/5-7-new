"""
Модуль для реализации RAG (Retrieval-Augmented Generation).

RAG объединяет поиск релевантной информации (Retrieval) с генерацией ответа (Generation)
для создания более точных и информативных ответов на вопросы пользователя.
"""

from typing import List, Tuple, Optional
import html
import os
from urllib.parse import quote

from openai_client import create_openai_client

# Названия документов базы знаний (ключ = имя txt-файла без расширения)
DOCUMENT_TITLES = {
    "poli": "СП 29.13330.2011 «Полы»",
    "organizacia stroitelstva": "СП 48.13330.2011 «Организация строительства»",
    "ograzhdausie konstrukcii": "СП 70.13330.2012 «Несущие и ограждающие конструкции»",
}


def format_source_references(
    search_results: List[Tuple[str, str, float]],
    html: bool = False,
) -> str:
    """
    Форматирует список источников для ответа пользователю.

    Если задана переменная DOCS_BASE_URL, добавляет кликабельные ссылки на файлы.
    """
    if not search_results:
        return ""

    base_url = os.getenv("DOCS_BASE_URL", "").rstrip("/")
    lines = ["📚 Источники:"]
    seen = set()

    for chunk_text, source, distance in search_results:
        if source in seen:
            continue
        seen.add(source)

        title = DOCUMENT_TITLES.get(source, source.replace("_", " "))
        relevance = max(0, min(100, int((1 - distance) * 100)))
        snippet = " ".join(chunk_text.split())[:120]
        if len(chunk_text) > 120:
            snippet += "…"
        filename = f"{source}.txt"

        if base_url and html:
            url = f"{base_url}/{quote(filename)}"
            lines.append(
                f'• <a href="{url}">{html.escape(title)}</a> ({relevance}%)\n'
                f"  <i>{html.escape(snippet)}</i>"
            )
        elif base_url:
            url = f"{base_url}/{quote(filename)}"
            lines.append(f"• {title} ({relevance}%)\n  {url}\n  «{snippet}»")
        else:
            lines.append(
                f"• {title} — docs/{filename} ({relevance}%)\n  «{snippet}»"
            )

    return "\n".join(lines)


class RAGAssistant:
    """
    Класс RAG-ассистента, который использует векторный поиск и LLM для ответов.
    
    Процесс работы:
    1. Получает запрос пользователя
    2. Ищет релевантные документы в векторной базе
    3. Формирует контекст из найденных документов
    4. Отправляет запрос + контекст в LLM
    5. Возвращает сгенерированный ответ
    """
    
    def __init__(
        self, 
        embedding_store,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Инициализация RAG-ассистента.
        
        Args:
            embedding_store: Экземпляр EmbeddingStore для поиска документов
            api_key: API ключ OpenAI (если None, берется из переменной окружения)
            model: Название модели OpenAI для генерации ответов
            temperature: Параметр "креативности" модели (0.0 - детерминированный, 1.0 - креативный)
        """
        self.embedding_store = embedding_store
        self.model = model
        self.temperature = temperature
        
        # Инициализируем клиент OpenAI
        # API ключ берется из параметра или переменной окружения OPENAI_API_KEY
        self.client = create_openai_client(api_key)
        
        print(f"✓ RAG-ассистент инициализирован (модель: {model})")
    
    def _format_context(self, search_results: List[Tuple[str, str, float]]) -> str:
        """
        Форматирует результаты поиска в контекст для LLM.
        
        Args:
            search_results: Список результатов поиска (текст, источник, расстояние)
            
        Returns:
            Отформатированный текст контекста
        """
        if not search_results:
            return "Релевантных документов не найдено."
        
        context_parts = []
        
        for i, (chunk_text, source, distance) in enumerate(search_results, 1):
            context_parts.append(
                f"[Документ {i} - {source}]\n{chunk_text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Создает промпт для LLM, включающий контекст и запрос пользователя.
        
        Args:
            query: Запрос пользователя
            context: Контекст из найденных документов
            
        Returns:
            Сформированный промпт
        """
        prompt = f"""Ты - полезный AI-ассистент. Используй следующую информацию из базы знаний, чтобы ответить на вопрос пользователя.

ВАЖНО: 
- Отвечай на основе предоставленного контекста
- Если в контексте нет информации для ответа, честно скажи об этом
- Отвечай на русском языке
- Будь конкретным и информативным

=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{query}

=== ОТВЕТ ===
"""
        return prompt
    
    def generate_response(
        self, 
        query: str, 
        top_k: int = 3,
        verbose: bool = True
    ) -> Tuple[str, List[Tuple[str, str, float]]]:
        """
        Генерирует ответ на запрос пользователя используя RAG.
        
        Это основной метод, который:
        1. Ищет релевантные документы
        2. Формирует контекст
        3. Отправляет запрос в LLM
        4. Возвращает ответ
        
        Args:
            query: Запрос пользователя
            top_k: Количество документов для поиска
            verbose: Выводить ли детальную информацию о процессе
            
        Returns:
            Кортеж (ответ_llm, список_найденных_документов)
        """
        # Шаг 1: Поиск релевантных документов в векторной базе
        if verbose:
            print(f"\n🔍 Поиск релевантных документов (top_k={top_k})...")
        
        search_results = self.embedding_store.search(query, top_k=top_k)
        
        if verbose and search_results:
            print(f"\n📚 Найдено {len(search_results)} релевантных фрагментов:")
            for i, (chunk, source, distance) in enumerate(search_results, 1):
                print(f"  {i}. [{source}] (similarity: {1 - distance:.3f})")
                print(f"     {chunk[:100]}...")
        
        # Шаг 2: Форматируем контекст из найденных документов
        context = self._format_context(search_results)
        
        # Шаг 3: Создаем промпт с контекстом и запросом
        prompt = self._create_prompt(query, context)
        
        # Шаг 4: Отправляем запрос в LLM
        if verbose:
            print(f"\n🤖 Генерация ответа с помощью {self.model}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты - полезный AI-ассистент, который отвечает на вопросы на основе предоставленного контекста."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # Извлекаем текст ответа
            answer = response.choices[0].message.content.strip()
            
            return answer, search_results
            
        except Exception as e:
            error_message = f"Ошибка при генерации ответа: {str(e)}"
            print(f"❌ {error_message}")
            return error_message, search_results
    
    def simple_response(self, query: str) -> str:
        """
        Упрощенная версия generate_response, возвращающая только текст ответа.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Ответ LLM
        """
        answer, _ = self.generate_response(query, verbose=False)
        return answer

