# RAG-Based Customer Service Chatbot using Gemini

A production-ready Retrieval-Augmented Generation (RAG) chatbot powered by Google Gemini LLM for intelligent customer support in the smart home and electronics domain.

## 🎯 Project Overview

This project implements a complete RAG pipeline that:
- Retrieves relevant information from a knowledge base using vector similarity search
- Generates contextual, accurate responses using Google Gemini LLM
- Provides multiple interfaces (Streamlit UI, CLI, API)
- Includes comprehensive evaluation metrics

## 🏗️ Architecture

```
User Query → Embedding → Vector Search (FAISS) → Context Retrieval 
                                                         ↓
                                           Prompt Engineering
                                                         ↓
                                            Gemini LLM Generation
                                                         ↓
                                              Response to User
```

## 📁 Project Structure

```
rag_customer_chatbot/
│
├── data/
│   ├── raw/
│   │   └── kb.txt                  # Knowledge base (place your data here)
│   ├── processed/
│   │   └── chunks.json             # Processed chunks
│   └── vector_store/
│       └── faiss_index/            # FAISS index files
│
├── src/
│   ├── config.py                   # Configuration and settings
│   ├── data_loader.py              # Data loading utilities
│   ├── text_preprocessing.py       # Text cleaning and normalization
│   ├── chunking.py                 # Document chunking logic
│   ├── embeddings.py               # Gemini embedding generation
│   ├── vector_store.py             # FAISS vector store management
│   ├── retriever.py                # Context retrieval logic
│   ├── prompt_templates.py         # Prompt engineering templates
│   ├── llm_gemini.py               # Gemini LLM wrapper
│   └── rag_pipeline.py             # End-to-end RAG pipeline
│
├── app.py                          # Streamlit web interface
├── cli_app.py                      # Command-line interface
├── build_index.py                  # Index building script
├── evaluate.py                     # Evaluation script
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag_customer_chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-pro
EMBEDDING_MODEL=models/embedding-001
```

4. **Prepare your knowledge base**

Place your knowledge base text file at:
```
data/raw/kb.txt
```

The provided knowledge base contains smart home product information, policies, and troubleshooting guides.

5. **Build the vector index**
```bash
python build_index.py
```

This will:
- Load and preprocess the knowledge base
- Chunk the documents
- Generate embeddings using Gemini
- Create and save the FAISS vector index

## 💻 Usage

### Streamlit Web Interface (Recommended)

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

Features:
- Interactive chat interface
- View retrieved context
- See relevance scores
- Clear conversation history
- System statistics

### Command-Line Interface

```bash
python cli_app.py
```

Commands:
- Type your question and press Enter
- `quit` or `exit` - Exit the application
- `clear` - Clear conversation history
- `stats` - Show system statistics

### Evaluation

Run the evaluation script to test the chatbot:

```bash
python evaluate.py
```

This will:
- Test the chatbot with sample questions
- Measure response accuracy
- Calculate latency metrics
- Generate a detailed evaluation report

## 📊 Evaluation Metrics

The system tracks:
- **Response Accuracy**: Keyword presence in responses
- **Contextual Relevance Score**: Similarity scores from vector search
- **Latency**: Average response time
- **User Satisfaction Score**: Based on simulated interactions
- **Precision & Recall**: For retrieval system evaluation

## 🎓 Skills Demonstrated

This project showcases:
- ✅ RAG pipeline implementation
- ✅ Google Gemini LLM integration
- ✅ Vector database management (FAISS)
- ✅ Natural Language Processing
- ✅ Prompt engineering
- ✅ Text similarity search
- ✅ API integration
- ✅ Full-stack application development (Streamlit)
- ✅ Performance evaluation and metrics

## 🔧 Configuration

Edit `src/config.py` or `.env` file to customize:

- `CHUNK_SIZE`: Size of text chunks (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K_RESULTS`: Number of chunks to retrieve (default: 3)
- `MAX_TOKENS`: Maximum response length (default: 1024)
- `TEMPERATURE`: LLM creativity (default: 0.7)

## 📝 Sample Questions

Try these questions with the chatbot:

1. **Policy Questions**
   - "What is the return policy for small electronics?"
   - "How do I cancel my order?"
   - "What payment methods do you accept?"

2. **Product Information**
   - "Tell me about the Smart Refrigerator"
   - "What are the specifications of the washing machine?"
   - "How much does the Smart Thermostat cost?"

3. **Technical Support**
   - "How do I set up the Smart Thermostat?"
   - "My security camera won't connect to WiFi"
   - "How do I fix washing machine vibration?"

## 🐛 Troubleshooting

### Index Not Found Error
```bash
# Rebuild the index
python build_index.py
```

### API Key Error
- Ensure your `.env` file contains a valid `GOOGLE_API_KEY`
- Check that the API key has access to Gemini models

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📈 Future Enhancements

- [ ] Add support for multiple knowledge bases
- [ ] Implement user authentication
- [ ] Add conversation memory across sessions
- [ ] Integrate with external APIs (order tracking, etc.)
- [ ] Add multi-language support
- [ ] Implement feedback collection system
- [ ] Deploy as REST API
- [ ] Add A/B testing framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Vignesh A

## 🙏 Acknowledgments

- Google Gemini API for LLM capabilities
- FAISS library for efficient vector search
- Streamlit for the web interface
- The open-source community


