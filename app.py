import os
import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI

# Настройка Streamlit
st.set_page_config(page_title="Строительный ИИ-ассистент", page_icon="🏗")
st.title("🏗 ИИ помощник по нормативке")

# Настройка OpenAI
# Нужно установить переменную окружения OPENAI_API_KEY с вашим ключом
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_FOLDER = "docs"

# Загружаем PDF
def load_pdfs(folder):
    pdf_texts = []
    if not os.path.exists(folder):
        st.error(f"Папка '{folder}' не найдена")
        return pdf_texts
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            pdf_texts.append((file, text))
    return pdf_texts

pdf_texts = load_pdfs(PDF_FOLDER)

# Поле для запроса
question = st.text_input("Задай вопрос:")

# Поиск и генерация ответа через GPT
def ask_gpt(question, documents):
    # Объединяем тексты документов (можно брать только релевантные куски для оптимизации)
    context = "\n\n".join([f"Документ: {doc[0]}\n{doc[1][:2000]}" for doc in documents])  # первые 2000 символов
    prompt = f"Ты строительный эксперт. Используя следующие документы:\n{context}\n\nОтветь на вопрос: {question}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    answer = response.choices[0].message.content
    return answer

if question and pdf_texts:
    with st.spinner("Обрабатываю вопрос..."):
        answer = ask_gpt(question, pdf_texts)
    st.markdown(f"**Ответ ИИ:**\n{answer}")
elif question:
    st.warning("Нет загруженных PDF для анализа")
