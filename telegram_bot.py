"""
Модуль для Telegram бота, интегрированного с RAG-ассистентом.

Бот позволяет пользователям задавать вопросы ассистенту через Telegram
и получать ответы на основе векторного поиска и LLM.
"""

import os
import time
from typing import Optional
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from rag import RAGAssistant, format_source_references
from cache import ResponseCache
from db_logger import DatabaseLogger
from http_proxy import get_proxy_url
from telegram.request import HTTPXRequest
from telegram.constants import ParseMode

DEMO_MESSAGE_LIMIT = 4
DEMO_LIMIT_MESSAGE = (
    "Демо-режим: 4 сообщения. Для открытия доступа свяжитесь с администратором."
)


class TelegramRAGBot:
    """
    Telegram бот для RAG-ассистента.
    
    Обрабатывает команды и сообщения от пользователей,
    логирует все взаимодействия в базу данных.
    """
    
    def __init__(
        self,
        token: str,
        rag_assistant: RAGAssistant,
        cache: ResponseCache,
        logger: DatabaseLogger
    ):
        """
        Инициализация Telegram бота.
        
        Args:
            token: Токен Telegram бота от @BotFather
            rag_assistant: Экземпляр RAG-ассистента
            cache: Экземпляр кеша ответов
            logger: Экземпляр логгера базы данных
        """
        self.rag_assistant = rag_assistant
        self.cache = cache
        self.logger = logger
        
        proxy = get_proxy_url()
        request = HTTPXRequest(
            proxy=proxy,
            httpx_kwargs={"trust_env": False},
        )
        get_updates_request = HTTPXRequest(
            proxy=proxy,
            httpx_kwargs={"trust_env": False},
            connection_pool_size=1,
        )

        # Создаем приложение Telegram
        self.application = (
            Application.builder()
            .token(token)
            .request(request)
            .get_updates_request(get_updates_request)
            .build()
        )
        
        # Регистрируем обработчики команд
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))
        
        # Регистрируем обработчик текстовых сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = """
🤖 Добро пожаловать в RAG-ассистента!

Я могу отвечать на ваши вопросы, используя базу знаний.

Доступные команды:
/help - показать справку
/stats - статистика системы
/logs - получить логи в CSV формате

Просто напишите мне вопрос, и я постараюсь на него ответить!

⚠️ Демо-режим: доступно до {limit} сообщений.
        """
        await update.message.reply_text(
            welcome_message.strip().format(limit=DEMO_MESSAGE_LIMIT)
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
📚 Справка по использованию бота:

• Просто напишите вопрос - я отвечу на основе базы знаний
• Использую RAG (Retrieval-Augmented Generation) для точных ответов
• Ответы кешируются для быстрой работы
• Демо-режим: до {limit} сообщений на пользователя

Команды:
/start - начать работу с ботом
/help - показать эту справку
/stats - статистика системы (документы, кеш)
/logs - получить логи взаимодействий в CSV формате

Примеры вопросов:
• "Что такое Python?"
• "Расскажи про RAG"
• "Что такое векторные базы данных?"
        """
        await update.message.reply_text(help_text.strip().format(limit=DEMO_MESSAGE_LIMIT))
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        try:
            # Получаем статистику системы
            doc_count = self.rag_assistant.embedding_store.collection.count()
            cache_size = self.cache.size()
            model = self.rag_assistant.model
            
            # Получаем статистику из логов
            log_stats = self.logger.get_stats()
            
            stats_message = f"""
📊 СТАТИСТИКА СИСТЕМЫ:

📚 База знаний:
  • Документов в ChromaDB: {doc_count}
  • Модель LLM: {model}

💾 Кеш:
  • Записей в кеше: {cache_size}

📝 Логи:
  • Всего запросов: {log_stats['total_requests']}
  • Из кеша: {log_stats['cached_requests']}
  • Уникальных пользователей: {log_stats['unique_users']}
  • Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс
            """
            
            await update.message.reply_text(stats_message.strip())
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при получении статистики: {str(e)}")
    
    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /logs - экспорт логов в CSV"""
        try:
            user_id = str(update.effective_user.id)
            
            # Экспортируем логи текущего пользователя
            csv_content = self.logger.export_to_csv(user_id=user_id)
            
            if not csv_content:
                await update.message.reply_text(
                    "📝 Логов для вашего пользователя не найдено."
                )
                return
            
            # Сохраняем во временный файл
            filename = f"logs_{user_id}_{int(time.time())}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            # Отправляем файл пользователю
            with open(filename, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption="📊 Ваши логи взаимодействий с ботом"
                )
            
            # Удаляем временный файл
            os.remove(filename)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при экспорте логов: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений от пользователей"""
        user_message = update.message.text
        user = update.effective_user
        user_id = str(user.id)
        username = user.username or user.first_name or "Unknown"

        if self.logger.get_telegram_dialog_count(user_id) >= DEMO_MESSAGE_LIMIT:
            await update.message.reply_text(DEMO_LIMIT_MESSAGE)
            return
        
        # Показываем, что бот печатает
        await update.message.chat.send_action(action="typing")
        
        start_time = time.time()
        
        try:
            search_results = []
            cached_answer = self.cache.get(user_message)
            from_cache = cached_answer is not None

            if cached_answer:
                answer = cached_answer
                search_results = self.rag_assistant.embedding_store.search(
                    user_message, top_k=3
                )
            else:
                answer, search_results = self.rag_assistant.generate_response(
                    query=user_message,
                    top_k=3,
                    verbose=False,
                )
                self.cache.set(user_message, answer)

            sources_text = format_source_references(
                search_results,
                html=bool(os.getenv("DOCS_BASE_URL")),
            )
            log_response = answer
            if sources_text:
                log_response = f"{answer}\n\n{sources_text}"
            
            # Вычисляем время ответа
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Логируем взаимодействие
            self.logger.log_interaction(
                query=user_message,
                response=log_response,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=from_cache,
                response_time_ms=response_time_ms
            )
            self.logger.increment_telegram_dialog_count(user_id)
            
            # Отправляем ответ пользователю
            max_length = 4000
            if len(answer) <= max_length:
                await update.message.reply_text(answer)
            else:
                parts = [answer[i:i + max_length] for i in range(0, len(answer), max_length)]
                for i, part in enumerate(parts):
                    await update.message.reply_text(part)

            if sources_text:
                parse_mode = ParseMode.HTML if os.getenv("DOCS_BASE_URL") else None
                await update.message.reply_text(
                    sources_text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True,
                )
            
            if from_cache:
                await update.message.reply_text("💾 (ответ из кеша)", quote=False)
        
        except Exception as e:
            error_message = f"❌ Произошла ошибка при обработке запроса: {str(e)}"
            await update.message.reply_text(error_message)
            
            # Логируем ошибку
            self.logger.log_interaction(
                query=user_message,
                response=error_message,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=False,
                response_time_ms=int((time.time() - start_time) * 1000)
            )
            self.logger.increment_telegram_dialog_count(user_id)
    
    def run(self):
        """Запускает бота"""
        print("🤖 Запуск Telegram бота...")
        print("Бот готов к работе! Нажмите Ctrl+C для остановки.")
        self.application.run_polling()

