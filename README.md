# FileWise: Empowering Insights, Effortlessly!

This repository contains code for a chatbot that can answer user questions based on the content of a file. The chatbot supports PDF, plain text, and DOCX file formats.

## Approach
The RAG module consists of two main phases: retrieval and generation. The retrieval phase retrieves relevant context from a knowledge document based on the user's question, and the generation phase uses a language model to generate a personalized answer using the retrieved knowledge. The goal is to create a chatbot that can accurately answer user questions from the provided knowledge document while preventing hallucination.

## Features

- Upload a file and ask questions about its content.
- Process PDF files using PyPDF2 library.
- Extract text from plain text and DOCX files using textract library.
- Split text into smaller chunks for efficient processing using CharacterTextSplitter from langchain library.
- Generate embeddings for text chunks using OpenAIEmbeddings from langchain library.
- Build a knowledge base of text chunks using FAISS from langchain library.
- Perform similarity search to find relevant documents based on user queries.
- Utilize a question-answering model to generate answers using load_qa_chain from langchain library.
- Display the generated answer to the user using Streamlit.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/tknishh/FileWise.git
```

2. Navigate to the project directory:

```bash
cd FileWise
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

***Note:*** Make sure to update your OpenAI API key in .env file.

4. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Open the application in your browser by visiting `http://localhost:8501` (or the address provided by Streamlit).
2. Click on the "Choose File" button to upload a file.
3. Once the file is uploaded, enter your question in the text input field.
4. The chatbot will process the file, search for relevant documents, and generate an answer.
5. The answer will be displayed below the text input field.

## Acknowledgements

This project utilizes the following libraries and frameworks:

- PyPDF2
- textract
- Streamlit
- langchain

## Assumptions
- The knowledge document contains sufficient information to answer user questions.
- The user questions are within the scope of the knowledge document.
- The chatbot will be a text-based interface.
- The chatbot will handle one user question at a time.

## Future Scope
- Improve retrieval performance by using more advanced models like DPR with passage re-ranking.
- Explore different generation techniques, such as controlled text generation or leveraging pretraining on domain-specific data.
- Enhance the chatbot's conversational abilities by incorporating dialogue management techniques and context tracking.
- Deploy the chatbot as a web application or integrate it into existing chat platforms.
- Incorporate feedback loops to continuously improve the chatbot's performance and address user queries.
- Expand the knowledge base and keep it up to date with the latest information.

## Contact

For any inquiries, please email [tanishqkhandelwaltqk011@gmail.com](mailto:tanishqkhandelwaltqk011@gmail.com).
