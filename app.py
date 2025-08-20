import os
import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="Строительный ассистент", page_icon="🏗")
st.title("🏗 Поиск по PDF-документам")

# Папка с PDF
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

# Поиск по PDF
def search_in_pdfs(query, documents):
    results = []
    query_lower = query.lower()
    for filename, text in documents:
        if text and query_lower in text.lower():
            # Находим первые 500 символов с вхождением
            idx = text.lower().find(query_lower)
            start = max(0, idx - 100)
            end = min(len(text), idx + 400)
            snippet = text[start:end].replace("\n", " ")
            results.append((filename, snippet))
    return results

if question and pdf_texts:
    results = search_in_pdfs(question, pdf_texts)
    if results:
        st.success(f"Найдено {len(results)} совпадений:")
        for res in results:
            st.write(f"📄 В документе **{res[0]}**:\n{res[1]}")
    else:
        st.warning("❌ Ничего не найдено в документах")
elif question:
    st.warning("Нет загруженных PDF для анализа")
