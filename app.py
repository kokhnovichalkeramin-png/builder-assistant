import os
import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="Строительный ассистент", page_icon="🏗")
st.title("🏗 ИИ помощник по нормативке")

# Поле для текста
question = st.text_input("Задай вопрос:")

# Папка с PDF
PDF_FOLDER = "docs"

# Получаем список PDF
if os.path.exists(PDF_FOLDER):
    pdf_paths = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
else:
    st.error(f"Папка '{PDF_FOLDER}' не найдена")
    pdf_paths = []

def search_in_pdfs(query, pdfs):
    results = []
    for pdf_path in pdfs:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text and query.lower() in text.lower():  # проверяем, что text не None
                results.append((pdf_path, text[:500]))  # показываем первые 500 символов
    return results

if question and pdf_paths:
    results = search_in_pdfs(question, pdf_paths)
    if results:
        for res in results:
            st.write(f"📄 Нашёл в {res[0]}: \n\n{res[1]}")
    else:
        st.write("❌ Ничего не нашёл в документах")
