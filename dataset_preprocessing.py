import pickle
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from datasets import load_from_disk
import datasets
from sklearn.model_selection import train_test_split
import torch
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
import random 

# remember this depends on the number of pickle files
total_job_number=50

data=[]
for i in range(0,total_job_number):
    filename = f"/mbtr_pickle_files/moleculeDic_{i}.pkl"
    with open(filename, 'rb') as f:
        while True:
            try:
                # Load the next pickled object
                unpickled_data = pickle.load(f)
                data.append(unpickled_data)
            except EOFError:
                # End of file reached
                break
            except pickle.UnpicklingError as e:
                print(f"Error: {e}")
                break
print(len(data))



random.shuffle(data)

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_dataset = Dataset.from_list(train_data)
test_dataset= Dataset.from_list(test_data)
validation_dataset=Dataset.from_list(val_data)

dataset={'train':train_dataset,'test':test_dataset,'validation':validation_dataset}

print(dataset)

# Create a dataset dictionary with a single dataset
mbtr_dataset = DatasetDict(dataset)

print(mbtr_dataset)

def mapping_func(dictionary, constant,prev_name,new_name):
    dataset=[]
    for value in dictionary:
        new_dict = {}
        new_value=[]
        for x in value[prev_name]:
            if x*constant>0:
                new_value.append(int(math.log((constant * x),2))+20)
            else:
                new_value.append(0)
        new_dict[new_name] = new_value
        dataset.append(new_dict)
        
    return dataset
def mapping_func_label(dictionary):
  temp_list=[]
  for value in dictionary:
    temp_dict={}
    temp_dict['labels']=float(value['homo_energy'])
    temp_list.append(temp_dict)
  return temp_list

train_x_list=mapping_func(mbtr_dataset["train"],10,'mbtr','input_ids')
train_x_dataset = Dataset.from_list(train_x_list)
train_y_list=mapping_func_label(mbtr_dataset["train"])
train_y_dataset = Dataset.from_list(train_y_list)
concatenated_train_dataset = Dataset.from_dict({"input_ids": train_x_dataset["input_ids"], "labels": train_y_dataset["labels"]})
print(concatenated_train_dataset)
print(concatenated_train_dataset['labels'][0])
print(concatenated_train_dataset['labels'][1])
print(concatenated_train_dataset['labels'][2])
test_y_list=mapping_func_label(mbtr_dataset["test"])
test_y_dataset = Dataset.from_list(test_y_list)
test_x_list=mapping_func(mbtr_dataset["test"],10,'mbtr','input_ids')
test_x_dataset = Dataset.from_list(test_x_list)
concatenated_test_dataset = Dataset.from_dict({"input_ids": test_x_dataset["input_ids"], "labels": test_y_dataset["labels"]})
print(concatenated_test_dataset)
val_y_list=mapping_func_label(mbtr_dataset["validation"])
val_y_dataset = Dataset.from_list(val_y_list)
val_x_list=mapping_func(mbtr_dataset["validation"],10,'mbtr','input_ids')
val_x_dataset = Dataset.from_list(val_x_list)
concatenated_val_dataset = Dataset.from_dict({"input_ids": val_x_dataset["input_ids"], "labels": val_y_dataset["labels"]})
dataset={'train':concatenated_train_dataset,'test':concatenated_test_dataset,'validation':concatenated_val_dataset}
prepared_dataset = DatasetDict(dataset)
prepared_dataset.save_to_disk("prepared_dataset")