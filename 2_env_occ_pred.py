"""
2_env_occ_pred.py
Author: Sin Yong Tan 2020-08-13
Updates by Maggie Jacoby
	2020-08-17: add argparser syntax and change save folders

This code takes in processed env csvs (full - all days) and outputs binary occupancy inferences

==== SY Notes ====
Input is expected to be cleaned / pre-processed beforehand
ie, no gaps/no missing data
(preferrable no missing timestamps too)
"""

# from DataLoader import DataLoader
# import numpy as np
import time
import json

import argparse
from utility_func import symbolize, state_gen, prob_thresh, inference

import pandas as pd
from pandas import read_csv
import os

from glob import glob
from natsort import natsorted






class Env_Pred(object):
	def __init__(self, json_path):

		with open(json_path, 'r') as f:
			d = json.load(f)
			self.name = d["sensor"]
			self.boundaries = d["boundaries"]
			self.TM = d['TM']
			self.threshold = d["threshold"]
			self.depth = d["depth"]
			self.tau = d["tau"]


	def process_data(self, data):
		syms = symbolize(data, self.boundaries)
		states = state_gen(syms, depth=self.depth, tau=self.tau)
		return states


	def occ_pred(self, states):
		occ_prob = inference(states, self.TM) # Inferencing
		pred = prob_thresh(occ_prob, self.threshold)
		return pred
		# return np.max(pred) # OR-gate: if any prediction is "1", then return "1"




def mylistdir(directory, bit='', end=True):
    filelist = os.listdir(directory)
    if end:
        return [x for x in filelist if x.endswith(f'{bit}') and not x.endswith('.DS_Store') and not x.startswith('Icon')]
    else:
         return [x for x in filelist if x.startswith(f'{bit}') and not x.endswith('.DS_Store') and not x.startswith('Icon')]
        
def make_storage_directory(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir






def main(data_path, model_path):
	# Loading model
	clf = Env_Pred(model_path) # call desired/designated json model

	# Loading data and perform SDF
	timestamped_data = read_csv(data_path, usecols=["timestamp",sensor])
	# @Maggie make this only read in non nan rows for the specifed sensor

	# start_time = time.time()
	data = clf.process_data(timestamped_data[sensor]) # get symbol and state

	# Inferencing:
	prediction = clf.occ_pred(data)
	# print(f"Time taken to infer: {time.time() - start_time} seconds")
	
	# Matching back the timestamp and the respective prediction
	prediction = pd.DataFrame(prediction) # last tau predictions is taken care of in the state_gen()
	prediction.columns = ["occupied"]
	timestamp = timestamped_data["timestamp"]
	timestamp = pd.DataFrame(timestamp[clf.depth:]).reset_index(drop=True) # Shifted due to state generation and tau

	timestamped_pred = timestamp.join(prediction)

	# Save path
	fname = os.path.basename(data_path) # output csv is named the same as the input csv, but saved to Inference_DB
# 	fname = f"H{H_num}_{station_color}S{station_num}_pred.csv"
# 	print(f"fname: {fname}")
	# save_folder = os.path.join(path,"Inference_DB",f"H{H_num}-{sta_col}",f"{station_color}S{station_num}","env")
	save_folder = os.path.join('/Users/maggie/Desktop/inferece_test', path,  'Inference_DB', hub)
	# save_folder = os.path.join(path, 'Inference_DB', hub)

	# if not os.path.exists(save_folder):
	# 	os.makedirs(save_folder)
	save_folder = make_storage_directory(save_folder)
	save_path = os.path.join(save_folder, fname)	
	timestamped_pred.to_csv(save_path,index=False)




if __name__ == '__main__':  
	model_location = "/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Inference-Environmental/env-Models"
	# Loading arg
	parser = argparse.ArgumentParser()
	parser.add_argument('-path','--path', default="AA", type=str, help='path of stored data')
	parser.add_argument('-hub', '--hub', default='', type=str, help='if only one hub... ')
	parser.add_argument('-save_location', '--save', default='', type=str, help='location to store files (if different from path')

	args = parser.parse_args()

	path = args.path
	save_path = args.save if len(args.save) > 0 else path
	home_system = path.strip('/').split('/')[-1]
	H = home_system.split('-')
	H_num, color = H[0], H[1][0].upper()
	hubs = [args.hub] if len(args.hub) > 0 else sorted(mylistdir(path, bit=f'{color}S', end=False))
	print(f'Hubs: {hubs}')
	for hub in hubs:

		sensor = 'temp_c'
		# @Maggie put loop in here to go through the modalities

		data_path = os.path.join(path, hub, 'processed_env', f'{H_num}_{hub}_full_cleaned.csv')

		model_path = glob(os.path.join(model_location, home_system, hub, 'env_model',f'{sensor}_*.json'))
		model_path = natsorted(model_path)[-1] # pick the last / mode complex model (p.s. natsorted is necessary, sorted doesn't sort the paths corectly)
		print(f'hub: {hub}, model_path: {os.path.basename(model_path)}')
		main(data_path=data_path, model_path=model_path)