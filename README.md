---

# Question Answering System with T5 Transformer

This project showcases the implementation of a question answering (QA) system using the powerful T5 transformer model.  It leverages the concept of transfer learning, first pretraining the model on a large text corpus (C4 dataset) to learn general language understanding and then fine-tuning it on a specialized question-answering dataset (SQuAD 2.0).

## Key Features

* **Transfer Learning:** Employs a two-step process:
    * **Pretraining:**  The model is first trained on the C4 dataset using Masked Language Modeling (MLM), enabling it to learn fundamental language patterns.
    * **Fine-tuning:**  The pretrained model is then fine-tuned on the SQuAD 2.0 dataset, specializing it for question answering tasks.
* **Transformer Architecture:** Utilizes & Built the state-of-the-art transformer architecture from scratch, known for its attention mechanisms that enable effective understanding of context and relationships within text.
* **TensorFlow & Keras:**  Built with TensorFlow and Keras, providing a flexible and efficient framework for deep learning model development.


## Step-by-Step Explanation (Question-Answering.ipynb):

1. **Import Libraries:** Import necessary Python libraries for data processing, model building, and visualization.

2. **Prepare Data (C4):**
   - Load C4 dataset samples (web pages) in JSON format.
   - Extract the 'text' field from each sample for pretraining.
   - Use `SentencePieceTokenizer` to tokenize the raw text into smaller units called subwords . (This is a subword tokenizer that splits 	words into meaningful fragments, ensuring a more efficient vocabulary. This step helps the model handle rare words and out-of-	vocabulary terms.)
   - Implement a `tokenize_and_mask` function to create input-target pairs for the MLM task. (This involves randomly masking some words in 	the input and expecting the model to predict them.)

3. **Pretrain T5 (C4):**
   - Define the transformer model architecture (number of layers, embedding size, etc.).
   - Create an optimizer and a loss function (`SparseCategoricalCrossentropy`).
   - Prepare the dataset for training by padding and batching tokenized sequences.
   - Train the model, where it learns to predict the masked words in the C4 dataset. This step is crucial, as it equips the model with 	general language understanding before we specialize it for QA.. (In practice, this would be done for many more epochs on the 	entire C4 dataset. Due to resource constraints, this notebook demonstrates the concept using a small subset of C4.)

4. **Fine-tune T5 (SQuAD 2.0):**
   - Load the SQuAD 2.0 dataset.
   - Parse the dataset to extract question-context-answer triplets. Create input sequences in the format "question: <Q> context: <C>" and 	target sequences in the format "answer: <A>."
   - Prepare training and testing data by tokenizing and padding.
   - Tokenize both the questions and contexts using the same `SentencePiece tokenizer` used for pretraining.
   - Fine-tune the pretrained T5 model on this SQuAD data. This process adjusts the model's internal parameters to make it exceptionally 	good at extracting answers from contexts given a question. (In a real-world scenario, this would also involve many more epochs.)
   - (Optional) Load a pretrained model to save time.

5. **Implement Question Answering:**
   - Define the `answer_question` function:
     - It takes a question and context as input.
     - Tokenize and pad the input question.
     - Initialize the answer with "answer: " token.
     - Iteratively predict the next word (predict the answer word by word) using the `transformer_utils.next_word` function.
     - Stop when the end-of-sequence (EOS) token is predicted.

6. **Test and Evaluate:**
   - Test the model on example questions from the SQuAD dataset.
   - Analyze the results to assess model performance.

**Key Transformer Concepts:**

* **Encoder:** Processes the input text (question + context) into a sequence of hidden representations, to understand the meaning of the question and its relationship to the context.
* **Decoder:** Generates the output text (answer) word by word, conditioned on the encoder's output and the previously generated words.
* **Attention:** Mechanism that allows the model to focus on relevant parts of the input while generating each word of the answer.
* **Positional Encoding:** Informs the model about the position of each word in the sequence, crucial for language understanding.

**Improvements:**

* **Data Augmentation:**  Use techniques like back-translation to increase the size and diversity of your training data.
* **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and model architectures to optimize performance.
* **Evaluation Metrics:** Use standard QA evaluation metrics like Exact Match (EM) and F1 score to measure the quality of your model's answers.
* **Error Analysis:** Analyze the types of errors the model makes to understand its limitations and areas for improvement. 


## Files

* `Question-Answering.ipynb`: Jupyter Notebook containing the main code for data processing, model creation, training, and evaluation.
* `transformer_utils.py`: Defines the transformer architecture from scratch and helper functions for training and inference.
* `utils.py`: Provides utility functions for tasks like positional encoding and token embedding.
* `data/`:  Directory containing the C4 and SQuAD datasets (ensure you have the necessary data files in this folder).
* `pretrained_models/`: (Optional) Directory to store pretrained model weights for future use.

## Usage

1. **Clone the Repository:** `git clone https://github.com/Shreyash-Gaur/T5_Question_Answering_System.git`
2. **Install Dependencies:** Install the required dependencies for the project.
3. **Prepare Data:** Download the C4 and SQuAD datasets and place them in the `data/` directory.
4. **Run the Notebook:** Open and execute the `Question-Answering.ipynb` notebook.
5. **Ask Questions:**  Use the `answer_question` function to pose questions to the model and obtain answers based on the context provided.

## Notes

* Due to computational constraints, the provided code trains the model on a limited portion of the C4 dataset and for a small number of epochs. 
* For optimal performance, consider training the model on the full C4 dataset and for a larger number of epochs using a more powerful environment (e.g., Google Colab).
* This project is intended for experiment purposes and demonstrates the core concepts of building a QA system with T5.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.

---

