import openai
from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from demo import OPENAI_API_KEY
from langchain_community.document_loaders import TextLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask_cors import CORS

openai.api_key = OPENAI_API_KEY


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}}, supports_credentials=True)




#=========== Login Endpoint =======================
users = {
    "testuser@11": "pwd11",
    "vansh@11": "van11",
}

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if username in users and users[username] == password:
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"message": "Invalid username or password"}), 401







#=========== Store Data Endpoint =======================
@app.route("/store", methods=["POST"])
def store_data():
    

    try:
        # Load the document
        loader = TextLoader("places.txt")
        doc = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_documents(doc)

        # Generate embeddings for the chunks
        texts = [chunk.page_content for chunk in chunks]
        embedding_func = OpenAIEmbeddings(api_key=openai.api_key)
        embeddings = embedding_func.embed_documents(texts)

        # Store the embeddings in ChromaDB
        embedding_dir = "chromadb_dir"
        dbb = Chroma.from_documents(chunks, embedding_func, persist_directory=embedding_dir)
        
        return jsonify({"message": f"{len(embeddings)} embeddings created and stored in ChromaDB"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500







#=========== Query Data Endpoint =======================
@app.route('/query', methods=['POST'])
def query_data():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        query = data.get('query')
        if not query:
            return jsonify({"error": "Query not provided"}), 400

        # Load the ChromaDB
        embedding_dir = "chromadb_dir"
        embedding_func = OpenAIEmbeddings(api_key=openai.api_key)
        dbb = Chroma(persist_directory=embedding_dir, embedding_function=embedding_func)

        # Query the database
        ans = dbb.similarity_search(query)
        context_txt = ans[0].page_content

        # Define the prompt template
        PROMPT_TEMPLATE = """ 
        Answer the question based only on the following context:
        {context}
        Answer the question based on the above context: {query}
        """
        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
        promptt = prompt_template.format(context=context_txt, query=query)

        response=openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": promptt}],
            max_tokens=60)

        answer = response.choices[0].message.content
        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500







#============= Initialize the Flask application =======================
if __name__ == '__main__':
    app.run(debug=True)
