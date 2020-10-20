import math
import torch
import subprocess
import itertools
import numpy as np 
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt 

def norm_sq_in_list(raw_list):
	return [torch.sum(tens**2) for tens in raw_list]

def compute_margins(output, target):
	target_val = output[range(target.size(0)), target]
	top_2_val, top_2_ind = output.topk(2, dim=1)
	best_non_target = top_2_val[:, 0]*(top_2_ind[:, 0] != target).float() +\
	top_2_val[:, 1]*(top_2_ind[:, 0] == target).float()
	margin_curr = target_val - best_non_target
	return margin_curr

def get_grad_hl_norms(hl_dict, loss, model, create_graph=False, only_inputs=True):
	key_list = hl_dict.keys()
	val_list = hl_dict.values()
	
	model.zero_grad()
	grad_list = torch.autograd.grad(
		loss,
		val_list,
		create_graph=create_graph,
		only_inputs=only_inputs,
    allow_unused=True)
	model.zero_grad()

	grad_norm_sq = norm_sq_in_list(grad_list)
	hl_norm_sq = norm_sq_in_list(val_list)

	norm_sq_dict = {key : (h_norm, g_norm) for (key, h_norm, g_norm) in zip(key_list, hl_norm_sq, grad_norm_sq)}

	return norm_sq_dict

class MeanDict():
	def __init__(self):
		self.mean_dict = {}
		self.counts_dict = {}

	def update(self, in_dict, count):
		for key, val in in_dict.items():
			if key not in self.mean_dict:
				self.counts_dict[key] = count
				self.mean_dict[key] = val
			else:
				curr_sum = self.counts_dict[key]*self.mean_dict[key] + in_dict[key]
				self.counts_dict[key] += count
				self.mean_dict[key] = curr_sum/self.counts_dict[key]

	def get_means(self):
		return self.mean_dict

# the type of regularization here
def update_step(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):
	if hparams["reg_type"] == "none":
		return none_update(
			criterion,
			optimizer,
			model,
			inputs,
			labels)
	if hparams["reg_type"] == "j_thresh":
		# only regularize the Jacobian
		return j_thresh_update(
			criterion,
			optimizer,
			model,
			inputs,
			labels,
			hparams)		

def none_update(
	criterion,
	optimizer,
	model,
	inputs,
	labels):
	start_time = time.time()
	output = model(inputs)
	loss = criterion(output, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss, output, time.time() - start_time

def j_thresh_update(
	criterion,
	optimizer,
	model,
	inputs,
	labels,
	hparams):
	start_time = time.time()
	output, hl_dict = model.forward_hl(inputs)

	loss = criterion(output, labels)
	margins = compute_margins(output, labels)

	norm_sq_dict = get_grad_hl_norms(hl_dict, torch.mean(margins), model, create_graph=True, only_inputs=True)
	reg_loss = 0
	for val in norm_sq_dict.values():
		j = val[1]
		j_ind = j > hparams["j_thresh"]
		if torch.sum(j_ind) > 0:
			reg_loss += hparams["data_reg"]*torch.mean(j[j_ind])

	full_loss = loss + reg_loss
	optimizer.zero_grad()
	full_loss.backward()
	optimizer.step()
	return loss, output, time.time() - start_time