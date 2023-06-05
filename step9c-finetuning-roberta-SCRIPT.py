import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaForSequenceClassification


print(torch.cuda.is_available())




# Load fine tuning data 1.3k manually labelled zoom tweets
df = pd.read_csv('zoom-sentiment-finetuning-sheet.csv')


# checking data and any missing values 
# missing values will stop the sentiment analysis
nan_rows = df[df['sentiment_combined_final_label'].isna()]
print("Number of rows with NaN values in 'label':", len(nan_rows))
print("Indices of rows with NaN values in 'label':", nan_rows.index)
print(df.shape)

#dropping all nan rows
df = df.dropna(subset=['sentiment_combined_final_label'])
print(f"df.shape after dropna: {df.shape} ")


# getting the tweets and labels ready as lists for the fine tuning
tweets = df['clean_tweet'].tolist()
labels = df['sentiment_combined_final_label'].tolist()  # I assume your labels are integers



num_labels = 3

# Load pre-trained model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


# Tokenize tweets
encodings = tokenizer(tweets, truncation=True, padding=True, max_length=512)



# Prepare dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = TweetDataset(encodings, labels)



from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        return inputs

    def __len__(self):
        return len(self.texts)

train_dataset = MyDataset(tweets, labels)

val_dataset = MyDataset(tweets, labels)


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics=compute_metrics,     
)



# Train the model
trainer.train()




# Save the fine-tuned model
model.save_pretrained("SAVED_ROBERTA_finetuned_model_CUDA")



