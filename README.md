# Text-Similarity-Search
## Description
Text-Similarity-Search is a Python application that allows users to perform similarity search on a dataset using different similarity measures. The application utilizes the SentenceTransformers library for encoding text and calculating similarity scores.

## Installation
To install and run the Text-Similarity-Search application, follow these steps:

## Clone the repository:
```bash
git clone https://github.com/your-username/Text-Similarity-Search.git
```
Navigate to the project directory:
```bash
Copy code
cd Text-Similarity-Search
```
Install the required libraries:
```bash
pip install -r requirements.txt
```
Run the application:
```bash
streamlit run app.py
```

## Libraries Used
The Text-Similarity-Search application utilizes the following libraries:
1. streamlit: For building the user interface and running the application.
2. pinecone: For performing similarity search using the Pinecone service.
3. pandas: For data manipulation and preprocessing.
4. numpy: For numerical operations.
5. sentence_transformers: For text encoding and similarity calculation.
6. sklearn: For cosine similarity calculation.

## Approach
The Text-Similarity-Search application follows the following approach:
* Load the dataset containing movie information.
* Preprocess the dataset by removing missing values and resetting the index.
* Initialize the SentenceTransformer model for text encoding.
* Load precomputed query vectors.
* Initialize the Pinecone service and create an index.
* Define a function for performing cosine similarity search on the dataset.
* Define a function for performing similarity search using Pinecone.
* Create a Streamlit user interface with an input for query and a selection for similarity measure.
* On button click, perform similarity search based on the selected measure and display the results.
* Steps to Improve

### Here are some steps to improve the Text-Similarity-Search application:
- [ ] Implement a more advanced text encoding model to capture semantic information better.
- [ ] Use a larger and more diverse dataset for better search results.
- [ ] Implement additional similarity measures to provide more options to the users.
- [ ] Add support for custom dataset upload and indexing.
- [ ] Optimize the search process for faster response times.
- [ ] Improve the user interface with better styling and interactive features.
- [ ] Add error handling and validation for user inputs.
- [ ] Implement unit tests to ensure the correctness of the application.
- [ ] Provide documentation and examples for easier usage and understanding.

## License
This project is licensed under the MIT License.
