import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import wandb



def load_and_process(dataset_name, artifact_name):

	run = wandb.init(project="Mestrado")
	artifact = run.use_artifact(artifact_name, type='dataset')
	artifact_dir = artifact.download()

	values = np.load(artifact_dir + "/" + dataset_name + ".npy", allow_pickle=True)


	df = pd.DataFrame(values, columns=['sigla_regiao', 'regiao', 'data', 'carga'])

	# Filter only from Sudeste
	df_sudeste = df[df['regiao'] == 'SUDESTE']

	# # Look for any duplicated values
	print(f'Duplicated values = {sum(df_sudeste.carga.duplicated())}\n')

	# # Preprocess
	df_sudeste = df_sudeste.drop(['sigla_regiao','regiao'], axis=1)
	df_sudeste = df_sudeste.dropna()
	df_sudeste = df_sudeste[df_sudeste.data.dt.year >= 2002]
	df_sudeste = df_sudeste.sort_values(by='data')
	df_sudeste = df_sudeste.reset_index()
	df_sudeste = df_sudeste.drop('index', axis=1)
	print(df_sudeste.head())

	draw_plot(df_sudeste)

	return df_sudeste



def draw_plot(df_sudeste):
	plt.figure(figsize=(24,7), dpi= 80)
	plt.plot('data', 'carga', data=df_sudeste, color='tab:red')

	# Decoration
	plt.yticks(fontsize=12, alpha=.7)
	xticks_labels = [str(x) for x in range(2002,2024)]
	xtick_location = pd.date_range('2001','2023', freq='Y')
	plt.xticks(ticks=xtick_location, labels=xticks_labels, rotation=45, fontsize=12, alpha=.7)
	plt.title("Carga elétrica diária - região sudeste (2002-2022)", fontsize=22)
	plt.grid(axis='both', alpha=.3)
	plt.xlabel('Ano',fontsize=16)
	plt.ylabel('MW', fontsize=16)

	# Remove borders
	plt.gca().spines["top"].set_alpha(0.0)    
	plt.gca().spines["bottom"].set_alpha(0.3)
	plt.gca().spines["right"].set_alpha(0.0)    
	plt.gca().spines["left"].set_alpha(0.3)   
	plt.show()


if __name__ == '__main__':

	dataset_name = "energy_demand"
	artifcat_name = "coimbra574/Mestrado/loaded_dataset:latest"
	path_to_preprocessed = "/home/taina/Desktop/Mestrado/Datasets/Dados_ONS"

	wandb.init(project="Mestrado")
	df = load_and_process(dataset_name, artifcat_name)
	wandb.finish()

	# Save preprocessed pickle dataframe
	df.to_pickle(path_to_preprocessed + '/' + 'energy_demand_preprocessed.pkl')

	# Remove all wandb files
	shutil.rmtree('wandb')
	shutil.rmtree('artifacts')





