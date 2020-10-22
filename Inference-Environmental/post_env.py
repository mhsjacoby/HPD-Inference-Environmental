"""
post_env.py
Authors: Sin Yong Tan and Maggie Jacoby
Edited: 2020-10-22 - convert occupied columns to numeric before taking max

Input: Folder with env inferences modality wise (10 seconds, all days)
Output: Day-wise CSVs with inferences on the 10 seconds for all modalitoes and combined

To run: python3 post_img.py -path /Volumes/TOSHIBA-18/H6-black/ 
	optional parameters: -hub, -start_date, -save_location
"""

import os
import sys
from glob import glob
import argparse

import numpy as np
import pandas as pd

import time
from datetime import datetime

from my_functions import *
from gen_argparse import *



if __name__ == '__main__':

	print(f'List of Hubs: {hubs}')

	for hub in hubs:
		print(f'hub: {hub}')

		file_paths = glob(os.path.join(path, 'Inference_DB', hub, 'env_by_mod', '*_inference_prob.csv'))
		fnames = [os.path.basename(file_path) for file_path in file_paths]
		sensors = [x.split('_')[2] + "_" + x.split('_')[3] for x in fnames]
		save_path = make_storage_directory(os.path.join(path, 'inference_DB', hub, 'env_inf'))

		all_files  = []
		for file_path in file_paths:
			sensor_name = os.path.basename(file_path).split('_')[2]
			data = pd.read_csv(file_path, index_col=0, names=['timestamp', f'occupied_{sensor_name}', f'prob_{sensor_name}'], dtype=object, header=0)
			all_files.append(data)
		del data
		all_data = pd.concat(all_files, axis=1)
		# all_data.to_csv('/Users/maggie/Desktop/all_data_full.csv')

		all_data.index = pd.to_datetime(all_data.index)
		start_date, end_date = all_data.index[0].date(), all_data.index[-1].date()
		timeframe = pd.date_range(start_date, end_date, freq='10S')[:-1]#.strftime('%Y-%m-%d %H:%M:%S')[:-1]

		# # # Complete the data
		all_data = all_data.reindex(timeframe, fill_value=np.nan)
		
		cols = [col for col in all_data.columns if 'occupied' in col]
		for col in cols:
			all_data[col] = pd.to_numeric(all_data[col])
		all_data['occupied'] = all_data[cols].max(axis=1)
		all_data["day"] = all_data.index.date
		unique_days = pd.unique(all_data["day"])

		for day in unique_days:
			data = all_data.loc[all_data['day'] == day]
			data = data.drop(columns=["day"])
			data.index.name = "timestamp"
			data.to_csv(os.path.join(save_path,f'{str(day)}.csv'))

