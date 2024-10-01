# Step 1: Install the necessary libraries


# Step 2: Import necessary libraries
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np

# Step 3: Load the CoNLL file you uploaded and convert it to a pandas DataFrame
def read_conll(file_path):
    sentences = []
    sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("-DOCSTART-") or line == "\n":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word_info = line.strip().split()
                if len(word_info) > 1:
                    sentence.append((word_info[0], word_info[-1]))  # Append word and its entity tag
        if sentence:
            sentences.append(sentence)
    
    return sentences

# File path to your CoNLL file (use the uploaded file path in your environment)
file_path = 'C:/Users/teeyob/Amharic_NER_for_E-commerce_Integration/data/labeled_data.conll'  # Adjust this if necessary for your file path
sentences = read_conll(file_path)

# Convert to pandas DataFrame
data = []
for sentence in sentences:
    for word, label in sentence:
        data.append([word, label])

df = pd.DataFrame(data, columns=['word', 'label'])
print(df.head())

# Step 4: Load pre-trained model (e.g., XLM-Roberta) and tokenizer
model_name = "xlm-roberta-base"  # You can replace with 'bert-tiny-amharic' or 'afroxmlr'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(df['label'].unique()))

# Step 5: Tokenize the data and align labels
# Step 5: Tokenize the data and align labels
def tokenize_and_align_labels(examples):
    # Debugging print
    print(f"Examples: {examples}")
    
    tokenized_inputs = tokenizer(examples["word"], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples["label"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words
        
        if word_ids is None:  # Check for None
            print(f"Warning: No word_ids for index {i}")
            continue
        
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get label -100
            elif word_idx != previous_word_idx:
                # If word_idx is in bounds of the label list, append the label; otherwise append -100
                try:
                    label_ids.append(label[word_idx])
                except IndexError:
                    label_ids.append(-100)
            else:
                # If a word is split into multiple tokens, assign -100 to subsequent tokens
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs




# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Step 6: Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

# Step 7: Data collator to pad the inputs
data_collator = DataCollatorForTokenClassification(tokenizer)

# Step 8: Load evaluation metric using the 'evaluate' library
metric = evaluate.load("seqeval")

# Step 9: Define function to compute evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label for label in label_ids if label != -100] for label_ids in labels]
    true_predictions = [
        [pred for pred, label in zip(prediction, label_ids) if label != -100]
        for prediction, label_ids in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Step 10: Fine-tune the model using Hugging Face's Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Step 11: Train the model
trainer.train()

# Step 12: Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
