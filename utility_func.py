import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from copy import deepcopy
import pandas as pd

def MEP(data, numparts):
	num_per_part = np.round(len(data)/numparts)
	data = np.sort(data)
	boundaries = []
	for i in range(1, numparts): # number of boundary line = numparts - 1
		idx = int(num_per_part*i)
		bound = np.mean([data[idx-1],data[idx]])
		boundaries.append(bound)
	return np.unique(boundaries)


def visualize_bounds(data, bounds):

	plt.figure()
	# plt.plot(data,'x')
	plt.plot(data)
	for j in bounds:
		plt.axhline(y=j, color='r', linestyle='-')
	
	# plt.title("")
	# plt.savefig('_partition.png')


def visualize_TM(TM, TM_title, show_prob=False):
	fig, ax = plt.subplots()
	plot = ax.imshow(TM, cmap=plt.cm.Blues)
	if show_prob:
		for i in range(np.shape(TM)[1]):
			for j in range(np.shape(TM)[0]):
				c = TM.values[j,i]
				ax.text(i, j, "%.2f"%(c), va='center', ha='center')
	plt.title(TM_title)
	plt.xlabel("state (t)")
	plt.ylabel("sym (t+1)")
	fig.colorbar(plot)


def visualize_pred(pred, truth):

	plt.figure()
	# plt.plot(data,'x')
	plt.plot(data,'x-')
	for j in bounds:
		plt.axhline(y=j, color='r', linestyle='-')
	
	# plt.title("")



def symbolize(data, boundaries):

	data = np.asarray(data)
# 	sym_data = np.zeros(np.shape(data))
	
	sym_data = np.asarray(["" for _ in range(len(data))])
# 	sym_data = np.chararray(len(data))

	if len(boundaries) == 1:
		sym_data[data <= boundaries[0]] = "1"
		sym_data[data > boundaries[0]] = "2"
	else:
		# deal with middle parts first, then lastly the lowest and highest portion
		for i in range(len(boundaries)-1):
			sym_data[((data > boundaries[i]) & (data <= boundaries[i+1]))] = str(i+2) # one-index symbols, but start with "2" for first symbol of middle portion
			
		# lowest and highest portion
		sym_data[data <= boundaries[0]] = "1" # lower portion
		sym_data[data > boundaries[-1]] = str(len(boundaries)+1) # upper portion

	return sym_data


def state_gen(sym_data, depth, tau):

	depth_sym_data = []

	# depth moving window
# 	for i in range(len(sym_data)-tau): # 自动去尾
	for i in range(len(sym_data)-depth-tau+1): # 自动去尾
# 		print(i)
		depth_sym_data.append(','.join(sym_data[i:i+depth])) # num sym = depth + 1

	return depth_sym_data




def compute_TM(state_data, occupancy):
	'''
	List unique state
	Find their index
	Create temp OCC list uing those index
	Compute sum of the temp occ list
	Divide by the temp list len to get only OCCUPIED probability
	'''
	state_data = np.asarray(state_data)
	occupancy = np.asarray(occupancy)

	TM = {}
	states = np.unique(state_data)

	for state in states:
		idx = np.argwhere(state_data == state)
		idx = [item for sublist in idx for item in sublist]
		temp_occ = occupancy[idx]
		occ_prob = np.sum(temp_occ)/len(temp_occ)
		TM[state] = occ_prob # only OCCUPIED probability 

	return TM

# Work for BINARY TM computed using both compute_TM() and pd.crosstab()
def inference(states, TM): 
	pred = []
	for index,state in enumerate(states):
		try:
			pred.append(TM[state])
			# print(f"State {state} Exist")
		except:
			print(f"State {state} DNE, forwarding previous prediction")
			pred.append(pred[-1]) # If state doesn't exist, take previous states's occ
	return pred
	# return [TM[state]  for state in states]


def thresh_opt(occ_prob, true_occ):

	threshold = np.unique(occ_prob) + 1e-8 # multiple thresholds
	temp_prediction = np.asarray(occ_prob)
	thresh_acc = []

	for thresh in threshold:
		temp_prediction_thresh = deepcopy(temp_prediction)
		
		index_1 = np.argwhere(temp_prediction > thresh)
		index_0 = np.argwhere(temp_prediction < thresh)
		
		temp_prediction_thresh[index_1] = 1
		temp_prediction_thresh[index_0] = 0

		temp = np.sum(abs(temp_prediction_thresh - true_occ))
		thresh_acc.append(1-(temp/len(true_occ)))

	# print best threshold:
	best_acc = np.amax(thresh_acc)
	print("max acc: %.4f"%(best_acc))
	best_acc_index = np.argwhere(np.asarray(thresh_acc) == best_acc)
	print("best_acc_index: ",best_acc_index[0][0])
	best_threshold = threshold[best_acc_index]
	best_threshold = best_threshold[0][0]
	print("best threshold: ",best_threshold)

	return best_threshold

def prob_thresh(occ_prob, thresh):

	occ_prob = np.asarray(occ_prob) # interested prediction prob. series
	pred = deepcopy(occ_prob) # placeholder for thresholded prediction
	
	index_1 = np.argwhere(occ_prob > thresh)
	index_0 = np.argwhere(occ_prob < thresh)
	
	pred[index_1] = 1
	pred[index_0] = 0


	return pred



def eval_(truth,pred):
	'''
	Returns dictionary containing accuracy, confusion matrix and tn_fp_fn_tp

	# Alternate method for accuracy:
	temp = np.sum(abs(pred - truth))
	acc = 1 - (temp/len(truth))
	'''
	print('===========================================================')
	acc = accuracy_score(truth,pred)*100

	cm = confusion_matrix(truth,pred)
	tn_fp_fn_tp = confusion_matrix(truth,pred).ravel()

	cm = np.asarray(cm)
	tn_fp_fn_tp = np.asarray(tn_fp_fn_tp)
	# print("tn_fp_fn_tp：",tn_fp_fn_tp)

	clf_rep = classification_report(truth,pred, target_names=['Unoccupied', 'Occupied']) # precision, recall, f1-score, support(num samples?)
	
	print(clf_rep)
	print("%.2f(%s)"%(acc,tn_fp_fn_tp)) # for ppt table record

	print('\n')

	# ================ Save Performances ================
	performance = {}
	performance['acc'] = acc
	performance['cm'] = cm
	performance['tn_fp_fn_tp'] = tn_fp_fn_tp
	performance['clf_rep'] = clf_rep

	# np.save(folder_path+'/H6_img_performances.npy',performance)
	return performance


def print_occ_percent(occ):

	# Check occupied and unoccupied
	occupied_index = np.argwhere(occ == 1)
	unoccupied_index = np.argwhere(occ == 0)

	# Data stat
	occ_percent = (len(occupied_index)/len(occ))*100
	unocc_percent = (len(unoccupied_index)/len(occ))*100
	print("%3.2f percent is occupied"%(occ_percent))
	print("%3.2f percent is unoccupied"%(unocc_percent))


