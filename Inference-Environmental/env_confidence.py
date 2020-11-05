"""
env_confidence.py
Author: Sin Yong Tan 2020-08-13
Updates by Maggie Jacoby
	2020-09-11: return probability prediction, as well as occupied/unoccupied

This code takes in processed env csvs (full - all days) and outputs binary occupancy inferences
previously called '2_env_occ_pred.py'

==== SY Notes ====
Input is expected to be cleaned / pre-processed beforehand
ie, no gaps/no missing data
(preferrable no missing timestamps too)
"""

# from DataLoader import DataLoader
# import numpy as np
import time
import json
import sys

# import argparse

from utility_func import symbolize, state_gen, prob_thresh, inference

import pandas as pd
from pandas import read_csv
import os

from glob import glob
from natsort import natsorted

from my_functions import *
from gen_argparse import *




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
		return pred, occ_prob
		# return np.max(pred) # OR-gate: if any prediction is "1", then return "1"


def main(data_path, model_path):
	# Loading model
	clf = Env_Pred(model_path) # call desired/designated json model

	# Loading data and perform SDF
	timestamped_data = read_csv(data_path, usecols=["timestamp",sensor])
	data = clf.process_data(timestamped_data[sensor]) # get symbol and state

	# Inferencing:
	prediction, prob = clf.occ_pred(data)
	prediction = pd.DataFrame({'occupied': prediction, 'probability': prob}) # last tau predictions is taken care of in the state_gen()

	# prediction.columns = ["occupied", "Probability"]
	timestamp = timestamped_data["timestamp"]
	timestamp = pd.DataFrame(timestamp[clf.depth:]).reset_index(drop=True) # Shifted due to state generation and tau
	timestamped_pred = timestamp.join(prediction)

	# Save path
	fname = os.path.basename(f'{H_num}_{hub}_{sensor}_inference_prob.csv') # output csv is named the same as the input csv, but saved to Inference_DB
	save_folder = make_storage_directory(os.path.join(path, 'Inference_DB', hub, 'env_by_mod'))
	save_path = os.path.join(save_folder, fname)
		
	timestamped_pred.to_csv(save_path,index=False)
	print(f'writing CSV to {save_path} of length {len(timestamped_pred)}')




if __name__ == '__main__':  
	# sensors = ['temp_c', 'rh_percent', 'light_lux', 'co2eq_ppm']
	model_location = os.path.abspath(os.getcwd())
	print(f'List of Hubs: {hubs}')

	for hub in hubs:
		print(f'hub: {hub}')

		model_paths = glob(os.path.join(model_location, 'Models-env', home_system, hub, 'env_model', '*.json'))
		models = [os.path.basename(path_name) for path_name in model_paths]

		sensors = set([x.split('_')[0] + "_" + x.split('_')[1] for x in models])
		print(sensors) if len(sensors) > 0 else print(f'No models for hub {hub} in home {home_system}.')

		for sensor in sensors:
			print(f'sensor: {sensor}')
			data_path = os.path.join(path, hub, 'processed_env', f'{H_num}_{hub}_full_cleaned.csv')

			model_path = glob(os.path.join(model_location, 'Models-env', home_system, hub, 'env_model', f'{sensor}_*.json'))
			model_path = natsorted(model_path)[-1] # pick the last / mode complex model (p.s. natsorted is necessary, sorted doesn't sort the paths corectly)
			print(f'hub: {hub}, modaltiy: {sensor}, model used: {os.path.basename(model_path)}')
			main(data_path=data_path, model_path=model_path)

