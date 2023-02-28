import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import wandb


def load_df(path):
	df = pd.DataFrame()
	for file in os.listdir(path):
		data = pd.read_excel(path + '/' + file)
		df = pd.concat([df, data])
	return df


def process_and_save(df):
	df_sudeste = df[df['nom_subsistema'] == 'SUDESTE']

	# Find for any duplicated values
	sum(df_sudeste['din_instante'].duplicated())

	# Preprocess
	df_sudeste = df_sudeste.drop(['id_subsistema','nom_subsistema'], axis=1)
	df_sudeste = df_sudeste.rename(columns={'din_instante': 'data', 'val_cargaenergiamwmed': 'carga'})
	df_sudeste = df_sudeste.reset_index()
	df_sudeste = df_sudeste.drop('index', axis=1)
	df_sudeste = df_sudeste.dropna()
	df_sudeste = df_sudeste[df_sudeste['data'].dt.year >= 2002]
	df_sudeste = df_sudeste.sort_values(by='data')
	#draw_plot(df_sudeste)

	# Add preprocessed dataset to wandb
	np.save("energy_demand_sudeste.npy", np.array(df_sudeste))
	artifact = wandb.Artifact(name="preprocessed_dataset", type="dataset")
	artifact.add_file("energy_demand_sudeste.npy")




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

	wandb.init(project="Mestrado")

	dataset_path = '/home/taina/Desktop/Mestrado/Datasets/ONS/Dados ONS'

	df = load_df(dataset_path)
	process_and_save(df)
	wandb.finish()




