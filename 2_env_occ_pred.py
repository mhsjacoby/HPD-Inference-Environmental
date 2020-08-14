# from DataLoader import DataLoader
# import numpy as np
import time
import json

from utility_func import symbolize, state_gen, prob_thresh, inference

import pandas as pd
from pandas import read_csv
import os

from glob import glob
from natsort import natsorted

'''
Input is expected to be cleaned / pre-processed beforehand
ie, no gaps/no missing data
(preferrable no missing timestamps too)
'''

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


def main(data_path, model_path):
	# Loading model
	clf = Env_Pred(model_path) # call desired/designated json model

	# Loading data and perform SDF
	timestamped_data = read_csv(data_path, usecols=["timestamp",sensor])
	# start_time = time.time()
	data = clf.process_data(timestamped_data["temp_c"]) # get symbol and state

	# Inferencing:
	prediction = clf.occ_pred(data)
	# print(f"Time taken to infer: {time.time() - start_time} seconds")
	
	# Matching back the timestamp and the respective prediction
	prediction = pd.DataFrame(prediction[:-clf.tau]) # last tau predictions doesn't have a timestamp (in this csv)
	prediction.columns = ["occupied"]
	timestamp = timestamped_data["timestamp"]
	timestamp = pd.DataFrame(timestamp[clf.depth+clf.tau:]).reset_index(drop=True) # Shifted due to state generation and tau

	timestamped_pred = timestamp.join(prediction)

	# Save path
	fname = os.path.basename(data_path) # output csv is named the same as the input csv, but saved to Inference_DB
# 	fname = f"H{H_num}_{station_color}S{station_num}_pred.csv"
# 	print(f"fname: {fname}")
	save_path = os.path.join(drive_letter+":/","Inference_DB",f"H{H_num}-{sta_col}",f"{station_color}S{station_num}",fname)
# 	print(save_path)
	timestamped_pred.to_csv(save_path,index=False)




if __name__ == '__main__':  
	'''
	# Loading arg
	parser.add_argument('-drive','--drive_letter', default="AA", type=str, help='Hard Drive Letter')
	parser.add_argument('-H','--H_num', default='1', type=int, help='House number: 1,2,3,4,5,6')
	parser.add_argument('-sta_num','--station_num', default=1, type=int, help='Station Number')
	parser.add_argument('-sta_col','--station_color', default="B", type=str, help='Station Color')
	parser.add_argument('-sensor','--sensor', default="temp_c", type=str, help='temp_c,light_lux,rh_percent')

	args = parser.parse_args()

	drive_letter = args.drive_letter
	H_num = args.H_num
	station_num = args.station_num
	station_color = args.station_color
	sensor = args.sensor
# 	start_date_index = args.start_date_index
# 	end_date_index = args.end_date_index # Not implemented yet
	'''

	## Only for testing purposes
# 	'''
	drive_letter = "G"
	H_num = 1
	station_color = "R"
	station_num = 1
	sensor = "temp_c"
# 	'''

	# replace with dict
	if station_color == "B":
		sta_col = "black"
	elif station_color == "R":
		sta_col = "red"
	elif station_color == "G":
		sta_col = "green"

	# Needs Update
	# This needs to be updated for my format
	# currently for : data_path=r"G:\occ_detect_data\MAIN\H1-red\outlier_check\RS1\H1_RS1_self.csv"
	data_path = os.path.join(drive_letter+":/","occ_detect_data","MAIN",f"H{H_num}-{sta_col}","outlier_check",f"{station_color}S{station_num}",f"H{H_num}_{station_color}S{station_num}"+"_self.csv")

	model_path = os.path.join(drive_letter+":/","Inference_DB",f"H{H_num}-{sta_col}",f"{station_color}S{station_num}","env_model",f"{sensor}*")
	model_path = natsorted(glob(model_path))[-1] # pick the last / mode complex model
# 	(p.s. natsorted is necessary, sorted doesn't sort the paths corectly)

	
	# main(data_path=r"G:\occ_detect_data\MAIN\H1-red\outlier_check\RS1\H1_RS1_self.csv", model_path=r"G:\Inference_DB\H1-red\RS1\env_model\temp_c_s18_d6_t1_model.json")
	main(data_path=data_path, model_path=model_path)


