import openai
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Install required packages if you don't have them
# !pip install transformers language-index beautifulsoup4 requests scikit-learn

# Set up Sentence-BERT embeddings
model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

# Set up Language model for QA
language_model_name = "gpt2"  # Using gpt2 for demonstration since language may not be available on Hugging Face
language_tokenizer = AutoTokenizer.from_pretrained(language_model_name)
language_model = AutoModelForCausalLM.from_pretrained(language_model_name)

# Function to generate embeddings for a given text
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embedding_model(**tokens, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Use the last hidden state
    embeddings = hidden_states.mean(dim=1)
    return embeddings.detach().numpy().flatten()

# Function to perform Google search and retrieve top 5 webpages
def google_search(query, num_results=5):
    search_url = f"https://www.google.com/search?q={query}&num={num_results}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.Timeout:
        print("Google search request timed out.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Google search request failed: {e}")
        return []

    links = []
    for g in soup.find_all('div', class_='BNeawe UPmit AP7Wnd'):
        links.append(g.text)
        if len(links) == num_results:
            break
    print(f"Found {len(links)} URLs")
    return links

# Function to scrape content from a list of URLs
def scrape_webpages(urls):
    webpage_text = ""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                webpage_text += p.text + " "
            print(f"Successfully scraped {url}")
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return webpage_text

# 1. Input query, read context from file, retrieve webpages, create corpus
query = "What was the primary purpose of the Great Wall of China?"


# Read context from a txt file
with open('context.txt', 'r') as file:
    context = file.read()

# Debug print block
print("Starting the RAG QA process...")
print(f"Query: {query}")

# Perform Google search and scrape top 5 webpages
print("Performing Google search...")
urls = google_search(query)
print("Retrieved URLs:", urls)

print("Scraping webpages...")
webpage_context = scrape_webpages(urls)
print("Webpage content retrieved.")

combined_context = context + ' ' + webpage_context

# Chunking the combined context into smaller pieces
chunk_size = 200  # Assuming a max chunk length of 200 words
chunks = [combined_context[i:i + chunk_size] for i in range(0, len(combined_context), chunk_size)]
print(f"Total number of chunks created: {len(chunks)}")

# Pass chunks into the embedding model and get the vector DB
print("Generating embeddings for chunks...")
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
print("Generated embeddings for all chunks.")

# Set the number of neighbors to the minimum of k or the number of chunks available
k = min(3, len(chunks))

# Create NearestNeighbors model for similarity search
print("Creating NearestNeighbors model...")
neighbors = NearestNeighbors(n_neighbors=k, metric='euclidean')
neighbors.fit(np.array(chunk_embeddings))
print("NearestNeighbors model created.")

# Pass the query into the embedding model and get the query vector
print("Generating query embedding...")
query_vector = get_embedding(query)
print("Query embedding generated.")

# 2. Use query vector to search top k nearest neighbors from the vector DB
print("Performing NearestNeighbors search...")
distances, indices = neighbors.kneighbors([query_vector])
print("NearestNeighbors search completed.")

retrieved_contexts = [chunks[i] for i in indices[0]]

# 3. Form prompt using the retrieved context and the original query
retrieved_text = ' '.join(retrieved_contexts)
prompt_template = f"Below is a question and context related to it. Answer the question based on the context provided.\n\nQuestion: {query}\n\nContext: {retrieved_text}\n\nAnswer:"

# Generate response using language model
print("Generating response using language model...")
tokens = language_tokenizer(prompt_template, return_tensors='pt')
input_ids = tokens.input_ids
attention_mask = tokens.attention_mask

output_ids = language_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=200,
    num_return_sequences=1,
    do_sample=True,  # Enable sampling
    temperature=0.7,
    repetition_penalty=2.0
)
response = language_tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("LLM response received.")

# Output the response
print("\n\nGenerated Answer:")
print(response)
