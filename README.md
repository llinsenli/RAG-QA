# Retrieval-Augmented Generation (RAG) QA System

This project implements a Retrieval-Augmented Generation (RAG) QA system using [Sentence BERT](https://www.sbert.net/docs/sentence_transformer/training_overview.html#dataset) for embeddings and gpt2 as the language model for question answering. The system takes a query, reads context from a file, retrieves related content from the web, and combines these to provide an answer.

## Overview

The Retrieval-Augmented Generation (RAG) QA System expertly combines three key modules: the embedding model, vector database, and language model, to efficiently answer queries. It starts with the embedding model, specifically Sentence BERT, which generates deep contextual embeddings from both pre-loaded text and web-retrieved content. These embeddings are stored in a vector database, utilized for their quick retrieval based on query relevance through KNN searches. Finally, the language model interprets the combined context to deliver precise and relevant answers. 

## Project Structure

- `main.py`: Main script to run the RAG QA system.
- `context.txt`: Text file containing the initial context related to the query.
- `README.md`: Documentation for the project.


## How to Use

1. **Prepare the Context**: Add any relevant context information into `context.txt`. This context will be used along with the content retrieved from the web to answer the query.

2. **Run the Script**:

   ```sh
   python main.py
   ```

3. **Input Query**: The script contains a variable `query` where you can input the question you want answered.

4. **Output**: The answer to the query will be printed to the console.

## File Descriptions

- **context.txt**: This file should contain any text data that might be relevant to answering the query.
- **main.py**: The main script that reads the context, performs a Google search to retrieve relevant webpages, chunks the combined context, performs similarity search using FAISS, and generates the final answer using Llama.

## Functionality

1. **Google Search and Web Scraping**: The script will perform a Google search to find the top 5 webpages related to the query. It will then scrape the content of these webpages.

2. **Chunking and Embedding**: The context and scraped content are combined, chunked into smaller pieces, and embedded using Sentence BERT.

3. **Similarity Search**: KNN is used to create a vector database from the embeddings and perform similarity search to retrieve the most relevant chunks.

4. **Answer Generation**: The query and retrieved context are used to generate a prompt, which is then passed to Llama for answer generation.

## Notes
- Ensure that you have access to the internet to perform Google searches and web scraping.
- The Google search might require adjustments depending on the region or Google blocking automation. You may need to use a different method to retrieve URLs if this becomes an issue.

## License
This project is licensed under the MIT License.