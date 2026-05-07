import os
import glob
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import chromadb
from openai import OpenAI

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

BOT_TOKEN = os.environ["BOT_TOKEN"]
DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
GROUP_ID = int(os.environ["GROUP_ID"])
MY_USER_ID = int(os.environ["MY_USER_ID"])

chroma = chromadb.PersistentClient(path="./chat_db")
collection = chroma.get_or_create_collection("messages")
llm = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")


def backfill_from_json(path="result.json"):
    """Разовая загрузка истории из экспорта Telegram."""
    if collection.count() > 0:
        logging.info(f"В базе уже {collection.count()} сообщений, пропускаю backfill")
        return
    if not os.path.exists(path):
        logging.warning(f"Файл {path} не найден — пропускаю backfill")
        return

    logging.info("Читаю экспорт чата...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs, metas, ids = [], [], []
    for m in data.get("messages", []):
        if m.get("type") != "message":
            continue
        text = m.get("text", "")
        if isinstance(text, list):
            text = "".join(
                t if isinstance(t, str) else t.get("text", "")
                for t in text
            )
        if not text.strip():
            continue
        docs.append(text)
        metas.append({
            "user": str(m.get("from", "unknown")),
            "date": str(m.get("date", "")),
        })
        ids.append(f"hist_{m['id']}")

    logging.info(f"Найдено {len(docs)} текстовых сообщений, индексирую...")
    BATCH = 1000
    for i in range(0, len(docs), BATCH):
        collection.add(
            documents=docs[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
            ids=ids[i:i+BATCH],
        )
        logging.info(f"  индексировано {min(i+BATCH, len(docs))}/{len(docs)}")
    logging.info("Backfill готов")


async def save_new_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Сохраняем каждое новое сообщение из целевой группы."""
    msg = update.message
    if not msg or not msg.text:
        return
    if msg.chat_id != GROUP_ID:
        return
    try:
        collection.add(
            documents=[msg.text],
            metadatas=[{
                "user": msg.from_user.full_name if msg.from_user else "unknown",
                "date": msg.date.isoformat(),
            }],
            ids=[f"live_{msg.chat_id}_{msg.message_id}"],
        )
    except Exception as e:
        logging.warning(f"Не удалось сохранить сообщение: {e}")


async def answer_question(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Отвечает в личке на вопросы — только разрешённому пользователю."""
    if update.effective_user.id != MY_USER_ID:
        await update.message.reply_text("Этот бот персональный.")
        return

    question = update.message.text
    await update.message.chat.send_action("typing")

    results = collection.query(query_texts=[question], n_results=20)
    if not results["documents"] or not results["documents"][0]:
        await update.message.reply_text("В базе пока нет сообщений.")
        return

    context_text = "\n".join(
        f"[{m['date'][:10]}] {m['user']}: {d}"
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    )

    try:
        response = llm.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты помощник, который отвечает на вопросы по сообщениям из группового чата. "
                        "Отвечай на основе предоставленных сообщений, указывай автора и дату когда уместно. "
                        "Если в сообщениях нет ответа — скажи об этом прямо, ничего не выдумывай."
                        ),
                    },
                {
                    "role": "user",
                    "content": (
                        f"Сообщения из чата:\n\n{context_text}\n\nВопрос: {question}"
                        ),
                    },
                ],
            )
        answer = response.choices[0].message.content
    except Exception as e:
        logging.error(f"Ошибка LLM: {e}")
        await update.message.reply_text(f"Ошибка: {e}")
        return

    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i:i+4000])


def main():
    backfill_from_json()
    app = Application.builder().token(BOT_TOKEN).build()
    # индексируем сообщения из группы
    app.add_handler(MessageHandler(
        filters.Chat(GROUP_ID) & filters.TEXT & ~filters.COMMAND,
        save_new_message,
    ))
    # отвечаем на вопросы в личке
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
        answer_question,
    ))
    logging.info("Бот запущен. Жду сообщения...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()