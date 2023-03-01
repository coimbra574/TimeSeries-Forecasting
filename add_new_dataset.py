import numpy as np
import pandas as pd
import os
import wandb


dataset_path = '/home/taina/Desktop/Mestrado/Datasets/Dados_ONS/Dados ONS'
dataset_name = "energy_demand"


df = pd.DataFrame()
for file in os.listdir(dataset_path):
	data = pd.read_excel(dataset_path + '/' + file)
	df = pd.concat([df, data])

# Add dataset to wandb
np.save(dataset_name + ".npy", np.array(df))
artifact = wandb.Artifact(name="loaded_dataset", type="dataset")
artifact.add_file(dataset_name + ".npy")

if os.path.exists(dataset_name + ".npy"):
	os.remove(dataset_name + ".npy") # one file at a time

wandb.init(project="Mestrado")
wandb.log_artifact(artifact)
wandb.finish()