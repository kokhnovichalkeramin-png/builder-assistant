# -*- coding: utf-8 -*-
# ================== 1) Импорты ==================
# Импортируем все необходимые библиотеки для работы.
import os
import pathlib
import gc
import streamlit as st # Используем Streamlit вместо Gradio
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama

# ================== 2) Пути ==================
# Путь к папке с вашими документами.
DOCS_DIR = "docs"

# Путь, где будет храниться числовой индекс ваших документов.
INDEX_DIR = "FAISS_Index"
os.makedirs(INDEX_DIR, exist_ok=True)

# ================== 3) Проверка и создание/загрузка индекса ==================
# Этот блок проверяет, создан ли уже индекс.
# Если нет — создает его, если есть — загружает.
if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
    st.info("Индекс не найден. Создаю новый...")
    
    # Загружаем документы из папки 'docs'
    loader = PyPDFDirectoryLoader(DOCS_DIR, recursive=True)
    all_docs = loader.load()

    if not all_docs:
        st.error("PDF не найдены. Положите файлы в папку 'docs'.")
        st.stop() # Останавливаем приложение, если нет документов

    # Разделяем документы на фрагменты для более точного поиска.
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    st.success(f"✅ Загружено страниц: {len(all_docs)} | Фрагментов: {len(chunks)}")

    # Загружаем модель для создания числовых представлений (эмбеддингов).
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Создаем индекс FAISS из фрагментов документов.
    vs = FAISS.from_documents(chunks, emb)

    # Сохраняем индекс, чтобы не создавать его заново при каждом запуске.
    vs.save_local(INDEX_DIR)
    del chunks
    gc.collect()
    
st.success("✅ Индекс готов!")

# Загружаем сохранённый индекс для использования в приложении.
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ================== 4) Загружаем локальную LLM (языковую модель) ==================
# Этот блок загружает "мозг" вашего чат-бота.
@st.cache_resource # Кэшируем модель, чтобы она не загружалась при каждом взаимодействии
def get_llm():
    return Llama.from_pretrained(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="*q4_k_m.gguf",
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=35, # Установите 0, если у вас нет GPU
        verbose=False,
    )

llm = get_llm()

# Инструкции для ИИ-модели, какой должна быть ее "личность".
SYSTEM_PROMPT = (
    "Ты помощник инженера-строителя в Республике Беларусь. "
    "Отвечай коротко (макс. 14-20 строк), строго по предоставленным фрагментам. "
    "Если нужного в фрагментах нет — прямо скажи, что в загруженных документах это не найдено. "
    "В конце добавь строку 'Источники:' со списком файлов и пунктов."
)

def build_context(docs, limit_chars=3500) -> Tuple[str, List[str]]:
    """Собирает контекст для языковой модели из найденных фрагментов."""
    ctx, cites, total = [], [], 0
    for d in docs:
        src = os.path.basename(d.metadata.get("source", ""))
        page = (d.metadata.get("page") or 0) + 1
        header = f"[{src}, стр. {page}]"
        text = (d.page_content or "").strip().replace("\u0000", " ").replace("\t", " ")
        if not text:
            continue
        snippet = text[:1200]
        ctx.append(f"{header}\n{snippet}")
        cites.append(header)
        total += len(snippet)
        if total >= limit_chars:
            break
    return "\n\n".join(ctx), cites

def rag_answer(question: str) -> str:
    """Основная функция для ответа на вопрос пользователя."""
    # 1) Ищем релевантные куски в документах.
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "В предоставленных файлах не найдено релевантного текста. Проверьте, что нужные PDF загружены."
    
    context, cites = build_context(docs)

    # 2) Формируем диалог для ИИ-модели.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Вопрос: {question}\n\nФрагменты из документов:\n{context}"}
    ]

    # 3) Генерируем ответ.
    output = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        top_p=0.9,
        max_tokens=320
    )
    text = output["choices"][0]["message"]["content"].strip()

    # Добавляем источники, если модель забыла их указать.
    if "Источники:" not in text:
        text += "\n\nИсточники:\n" + "\n".join(cites)

    return text

# ================== 5) Создаем интерфейс чата (Streamlit) ==================
# Заголовок страницы
st.set_page_config(page_title="Строительный ассистент", page_icon="🏗️")
st.title("🏗️ Строительный ассистент")
st.markdown("Поиск и ответы на вопросы по вашим документам.")

# Инициализируем историю чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображаем историю
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Обработка нового сообщения
if prompt := st.chat_input("Задай свой вопрос"):
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерируем ответ
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            response = rag_answer(prompt)
        st.markdown(response)
    
    # Добавляем ответ ассистента в историю
    st.session_state.messages.append({"role": "assistant", "content": response})

