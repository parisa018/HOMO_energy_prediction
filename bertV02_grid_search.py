#--------------------This file is used for grid search through hyperparameters-----------------------------
# use one overall saved dataset, 80%training, 10%test, 10% validation.
# we have plotting based on the training logs

from datasets import load_dataset
from datasets import load_from_disk
from transformers import BertConfig, BertModel
import pickle
from datasets import Dataset, DatasetDict
import datasets
from sklearn.model_selection import train_test_split
import torch
import math
import transformers
from transformers import DefaultDataCollator
from collections import defaultdict
import logging
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Read the outputs of MBTR   ----------------- prepare the dataset
prepared_dataset=load_from_disk('prepared_dataset')



# init the model

collated_data = DefaultDataCollator("pt")

class MyBertModel(torch.nn.Module):
     def __init__(self, config):
        super(MyBertModel,self).__init__()
        # vocab size depends on the number of the categories of MBTR output (number of bins accross y-axis)
        # embedding dim is user specified 
        # Embedding layer: vocab size x embedding dim
        self.embeddings = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim
        )
        self.bert = BertModel(config)
        # check different values of dropout
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        self.loss_fn = torch.nn.MSELoss()
        
        
     def forward(self, input_ids, labels=None, attention_mask=None):
        self.num_labels = config.num_labels
        outputs = self.bert(input_ids= input_ids, attention_mask=attention_mask)

        
        pooled_output = outputs['pooler_output']

        pooled_output = self.dropout(pooled_output)
        
        
        logits = self.classifier(pooled_output)
        
        
        logits=logits.squeeze(1)
        
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits
        
        
logging.disable(logging.INFO)
     
class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

                    
def compute_metrics_for_regression(outputs_and_labels):
    outputs, labels = outputs_and_labels
    labels = labels.reshape(-1, 1)
    
    mse = mean_squared_error(labels, outputs)
    mae = mean_absolute_error(labels, outputs)
    r2 = r2_score(labels, outputs)
    
    return {"mse": mse, "mae": mae, "r2": r2}

def plot(logs, keys, labels,learning_rate,batch_size):
    values = sum([logs[k] for k in keys], [])
    plt.clf()
    for key, label in zip(keys, labels):    
        plt.plot(logs["epoch"], logs[key], label=label)
    plt.legend()
    plt.suptitle('{} loss per epoch'.format(labels),fontsize=14, fontweight='bold')
    plt.title('learning rate= {}, batch size= {}'.format(learning_rate,batch_size))
    plt.savefig('{}_{}_{}.png'.format(labels,learning_rate,batch_size))
    plt.show()






learning_rate_list=[1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,2e-4,2e-5,2e-6,2e-7,2e-8,3e-4,3e-5,3e-6,3e-7,3e-8]
batch_size_list=[16,32,64,128,200]        



for lr in learning_rate_list:
    for batch_size in batch_size_list:
        
        print('batch size is {} and learning rate is {}'.format(batch_size,lr))
        
        
        # Define configuration for BERT model
        config = BertConfig(vocab_size=33,
                            hidden_size=144,
                            num_hidden_layers=8,
                            num_layers=8,
                            embedding_dim=64,
                            dropout_rate=0.1,
                            nonlinearity="relu")
        bert_model = MyBertModel(config)
        
        # Argument gives the number of steps of patience before early stopping
        early_stopping = transformers.EarlyStoppingCallback(
            early_stopping_patience=5
        )
        
        
        training_logs = LogSavingCallback()
        trainer_args = transformers.TrainingArguments(
            "model_checkpoints", #save checkpoints here
            evaluation_strategy="steps",
            logging_strategy="steps",
            eval_steps=500,
            logging_steps=500,
            learning_rate=lr,
            max_steps=20000,
            load_best_model_at_end=True,
            per_device_train_batch_size=batch_size
        )
    
    
        trainer = transformers.Trainer(
            model=bert_model,
            args=trainer_args,
            train_dataset=prepared_dataset['train'],
            eval_dataset=prepared_dataset['validation'], 
            compute_metrics=compute_metrics_for_regression,
            data_collator=collated_data,
            callbacks=[early_stopping, training_logs]
        )
    
        trainer.train()
    
        eval_results = trainer.evaluate(prepared_dataset['validation'])
        print(eval_results)
    
        for metric, value in eval_results.items():
            print(f"{metric}: {value}")
        
    
        plot(training_logs.logs, ["eval_loss","loss"], ["Evaluation loss","Training loss"],lr,batch_size)
        plot(training_logs.logs, ["eval_loss"], ["Evaluation loss"],lr,batch_size)