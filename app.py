import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="Строительный ассистент", page_icon="🏗")

st.title("🏗 ИИ помощник по нормативке")

# Поле для текста
question = st.text_input("Задай вопрос:")

# Заглушка для документов (позже подключим Google Drive)
pdf_paths = ["docs/sample.pdf"]  # временно

def search_in_pdfs(query, pdfs):
    results = []
    for pdf_path in pdfs:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if query.lower() in text.lower():
                results.append((pdf_path, text[:500]))
    return results

if question:
    results = search_in_pdfs(question, pdf_paths)
    if results:
        for res in results:
            st.write(f"📄 Нашёл в {res[0]}: \n\n{res[1]}")
    else:
        st.write("❌ Ничего не нашёл в документах")
