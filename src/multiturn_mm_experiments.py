#! /usr/bin/env python

import argparse
import pandas as pd
import torch
import os
import json
import time
import sys

from utilities.dataset_tasks_utils import ECQA, StrategyQA, GSM8k
from machine_teaching.models.model import UnidentifiedTaskError, UnidentifiedUtilityMetricError
from machine_teaching.models.hf.teacher_model_hf import TeacherModel as TeacherModelHF
from machine_teaching.models.vllm.teacher_model_vllm import TeacherModel as TeacherModelVLLM
from machine_teaching.models.hf.student_model_hf import StudentModel as StudentModelHF
from machine_teaching.models.vllm.student_model_vllm import StudentModel as StudentModelVLLM
from machine_teaching.models.hf.teacher_static_mental_model_hf import TeacherStaticMentalModel as TeacherMentalModelHF
from machine_teaching.models.vllm.teacher_static_mental_model_vllm import TeacherStaticMentalModel as TeacherMentalModelVLLM
from machine_teaching.models.vllm.teacher_dynamic_mental_model_vllm import TeacherDynamicMentalModel as TeacherDynamicMentalModelVLLM
from machine_teaching.error_prior.monitor import ErrorMonitor
from machine_teaching.error_prior.utility import ErrorUtilityMixin
from machine_teaching.error_prior.principles import PRINCIPLES
from functools import partial
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from vllm import LLM
from typing import Tuple, List, Optional, Dict, Union
from numpy.random import default_rng, Generator
from tqdm import tqdm
import copy


RNG_SEED = 25092024


class UnidentifiedLibError(Exception):
	"""Raise exception for a task not recognized."""
	pass


def get_teacher_model_samples(rng_gen: Generator, task: str, train_data: pd.DataFrame, student_samples: List[pd.Series], teacher_expl_type: str, num_samples: int,
							  student_model: Union[StudentModelHF, StudentModelVLLM], teacher_model: Union[TeacherModelHF, TeacherModelVLLM] = None) -> List[Dict]:
	
	teacher_samples = []
	
	if teacher_expl_type.find('blind') != -1:
		teacher_samples = student_samples
	
	elif teacher_expl_type.find('useful') != -1:
		print("Using useful teacher explanations")
		shuffle_train = train_data.sample(frac=1, random_state=rng_gen).reset_index(drop=True)
		idx = 0
		quit = False
		n_added = 0
		while not quit:
			sample = shuffle_train.iloc[idx].to_dict()
			
			student_prediction_no_intervene, _ = student_model.predict(sample, expl='', debug=False, intervene=False)  # get student prediction without teacher intervention
			
			teacher_expl = sample['explanation'] if teacher_model is None else teacher_model.predict(sample)[0]
			student_prediction_intervene, _ = student_model.predict(sample, expl=teacher_expl, debug=False, intervene=True)  # get student prediction with teacher intervention
			if task == 'ec_qa':
				student_prediction_no_intervene = int(student_prediction_no_intervene[0])
				student_prediction_intervene = int(student_prediction_intervene[0])
				sample['answer'] = int(sample['answer'])

			if student_prediction_intervene == sample['answer'] and student_prediction_no_intervene != student_prediction_intervene:  # add sample if the intervention made student right
				teacher_samples.append(sample)
				print("Sample:", sample)
				n_added += 1
			
			idx += 1
			quit = (len(teacher_samples) >= num_samples or idx >= shuffle_train.shape[0])
		print('Added %d samples to teacher context.' % n_added)
	
	else:
		samples_idxs = rng_gen.choice(train_data.shape[0], num_samples, replace=False)
		teacher_samples = [train_data.iloc[x].to_dict() for x in samples_idxs]
	
	return teacher_samples


def get_mental_model_samples(rng_gen: Generator, train_data: pd.DataFrame, task: str, mental_model_type: str, max_samples: int,
							 student_model: Union[StudentModelHF, StudentModelVLLM], teacher_model: Union[TeacherModelHF, TeacherModelVLLM] = None) -> Tuple[List, List]:
	
	shuffle_train = train_data.sample(frac=1, random_state=rng_gen).reset_index(drop=True)
	
	if task == 'strategy_qa':
		no_intervention_samples = [[], []]
		intervention_samples = [[], []]
		num_no_intervention_samples = [0, 0]
		num_intervention_samples = [0, 0]
		for row in shuffle_train.iterrows():
			
			# Break the cycle if reached the maximum number of samples
			if sum(num_no_intervention_samples) == max_samples or sum(num_intervention_samples) == max_samples:
				break
			
			sample = row[1].to_dict()
			
			# When you have either no intervention or both intervention modes mental model add no intervention samples
			if mental_model_type.find('no') != -1 or mental_model_type.find('both') != -1:
				student_prediction, student_explanation = student_model.predict(sample=sample, expl='', intervene=False)
				if ((student_prediction == 'yes' and num_no_intervention_samples[0] == max_samples // 2) or
						(student_prediction == 'no' and num_no_intervention_samples[1] == max_samples // 2)):      # keep number of yes and no ansewers balanced
					continue
				
				no_intervention_sample = {
						"question":            	sample['question'],
						"answer":              	sample['answer'],
						"explanation":    		sample['explanation'],
						"prediction":          	student_prediction,
						"student_explanation": 	student_explanation
				}
				
				if student_prediction == 'yes':
					no_intervention_samples[0].append(no_intervention_sample)
					num_no_intervention_samples[0] += 1
				else:
					no_intervention_samples[1].append(no_intervention_sample)
					num_no_intervention_samples[1] += 1
			
			# When you have either intervention or both intervention modes mental model add intervention samples
			if mental_model_type.find('inter') != -1 or mental_model_type.find('both') != -1:
				
				_, teacher_explanation = teacher_model.predict(sample=sample)
				student_prediction, student_explanation = student_model.predict(sample=sample, expl=teacher_explanation, intervene=True)
				if ((student_prediction == 'yes' and num_intervention_samples[0] == max_samples // 2) or
						(student_prediction == 'no' and num_intervention_samples[1] == max_samples // 2)):  # keep number of yes and no ansewers balanced
					continue
				
				intervention_sample = {
						"question":            	sample['question'],
						"answer":              	sample['answer'],
						"explanation":			sample['explanation'],
						"prediction":          	student_prediction,
						"teacher_explanation": 	teacher_explanation
				}
				
				if student_prediction == 'yes':
					intervention_samples[0].append(intervention_sample)
					num_intervention_samples[0] += 1
				else:
					intervention_samples[1].append(intervention_sample)
					num_intervention_samples[1] += 1
		
		intervention_samples = intervention_samples[0] + intervention_samples[1]
		no_intervention_samples = no_intervention_samples[0] + no_intervention_samples[1]
		rng_gen.shuffle(intervention_samples)
		rng_gen.shuffle(no_intervention_samples)
		
		return no_intervention_samples, intervention_samples
	
	elif task == 'ec_qa':
		no_intervention_samples = []
		intervention_samples = []
		num_no_intervention_samples = 0
		num_intervention_samples = 0
		
		for row in shuffle_train.iterrows():
			
			# Break the cycle if reached the maximum number of samples
			if num_no_intervention_samples == max_samples or num_intervention_samples == max_samples:
				break
			sample = row[1].to_dict()
			
			# When you have either no intervention or both intervention modes mental model add no intervention samples
			if mental_model_type.find('no') != -1 or mental_model_type.find('both') != -1:
				student_prediction, student_explanation = student_model.predict(sample=sample, expl='', intervene=False)
				
				no_intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"options":              sample['options'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"student_explanation":  student_explanation
				}
				
				no_intervention_samples.append(no_intervention_sample)
				num_no_intervention_samples += 1
			
			# When you have either intervention or both intervention modes mental model add intervention samples
			if mental_model_type.find('inter') != -1 or mental_model_type.find('both') != -1:
				
				_, teacher_explanation = teacher_model.predict(sample=sample)
				student_prediction, student_explanation = student_model.predict(sample=sample, expl=teacher_explanation, intervene=True)
				
				intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"options":              sample['options'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"teacher_explanation":  teacher_explanation
				}
				
				intervention_samples.append(intervention_sample)
				num_intervention_samples += 1
		
		return no_intervention_samples, intervention_samples
	
	elif task == 'gsm8k':
		no_intervention_samples = []
		intervention_samples = []
		num_no_intervention_samples = 0
		num_intervention_samples = 0
		
		for row in shuffle_train.iterrows():
			
			# Break the cycle if reached the maximum number of samples
			if num_no_intervention_samples == max_samples or num_intervention_samples == max_samples:
				break
			sample = row[1].to_dict()
			
			# When you have either no intervention or both intervention modes mental model add no intervention samples
			if mental_model_type.find('no') != -1 or mental_model_type.find('both') != -1:
				student_prediction, student_explanation = student_model.predict(sample=sample, expl='', intervene=False)
				
				no_intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"student_explanation":  student_explanation
				}
				
				no_intervention_samples.append(no_intervention_sample)
				num_no_intervention_samples += 1
			
			# When you have either intervention or both intervention modes mental model add intervention samples
			if mental_model_type.find('inter') != -1 or mental_model_type.find('both') != -1:
				
				_, teacher_explanation = teacher_model.predict(sample=sample)
				student_prediction, student_explanation = student_model.predict(sample=sample, expl=teacher_explanation, intervene=True)
				
				intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"teacher_explanation":  teacher_explanation
				}
				
				intervention_samples.append(intervention_sample)
				num_intervention_samples += 1
		
		return no_intervention_samples, intervention_samples
	
	else:
		raise UnidentifiedTaskError('Task %s not defined.' % task)


def load_models(rng_seed: int, train_data: pd.DataFrame, num_samples: int, student_model_path: str, teacher_model_path: str, task: str, use_explanations: bool, use_instruction: bool,
				student_expl_type: str, teacher_expl_type: str, mental_model_type: str, mm_dynamic: bool, intervention_utility: str, max_tokens: int, num_beams: int, cache_dir: Path, model_lib: str = 'hf',
				num_logprobs: int = 2, local_model: bool = True, s_model_url: str = '', t_model_url: str = '', api_key: str = '', temperature: float = 0.0) -> Tuple[Union[StudentModelHF, StudentModelVLLM], Optional[Union[TeacherModelHF, TeacherModelVLLM]], Optional[Union[TeacherMentalModelHF, TeacherMentalModelVLLM]]]:
	
	print('Using %s lib' % model_lib)
	rng_gen = default_rng(rng_seed)
	
	print('Setting up the Student Model')
	train_idxs = rng_gen.choice(train_data.shape[0], num_samples, replace=False)
	student_samples = [train_data.iloc[idx].to_dict() for idx in train_idxs]
	
	if model_lib == 'hf':

		if "llama" in student_model_path:
			student_gen_model = LlamaForCausalLM.from_pretrained(student_model_path, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
			student_tokenizer = LlamaTokenizer.from_pretrained(student_model_path, cache_dir=cache_dir, use_fast=False)
		else:
			student_tokenizer = AutoTokenizer.from_pretrained(student_model_path, cache_dir=cache_dir, use_fast=False)
			student_gen_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_path, device_map='auto', cache_dir=cache_dir)
			
		student_model = StudentModelHF(student_model_path, student_samples, student_gen_model, student_tokenizer, student_expl_type, task, max_tokens, num_beams, use_explanations)

		if use_explanations:
			print('Setting up the Teacher Model...')
			if student_expl_type.find('human') != -1:
				teacher_model = TeacherModelHF(teacher_model_path)
				mental_model = None
			else:
				if "llama" in teacher_model_path:
					teacher_gen_model = LlamaForCausalLM.from_pretrained(teacher_model_path, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.float16)
					teacher_tokenizer = LlamaTokenizer.from_pretrained(teacher_model_path, cache_dir=cache_dir, use_fast=False) if teacher_model_path != 'human' else None
				else:
					teacher_gen_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_path, device_map='auto', cache_dir=cache_dir)
					teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, cache_dir=cache_dir, use_fast=False) if teacher_model_path != 'human' else None
				
				print('Getting teacher samples')
				teacher_samples = get_teacher_model_samples(rng_gen, task, train_data, student_samples, teacher_expl_type, num_samples, student_model)
				print('Creating Teacher Model')
				teacher_model = TeacherModelHF(teacher_model_path, teacher_samples, teacher_gen_model, teacher_tokenizer, teacher_expl_type, task, max_tokens, num_beams, use_explanations)
				
				if intervention_utility.find('mm') != -1 or (intervention_utility.find('mental') != -1 and intervention_utility.find('model') != -1):
					print('Setting up the Teacher Mental Model...')
					print('Getting mental models samples...')
					mm_samples = get_mental_model_samples(rng_gen, train_data, task, mental_model_type, num_samples, student_model, teacher_model)
					print('Mental model samples retrieved.')
					print('Creating TeacherMentalModelHF...')
					mental_model = TeacherMentalModelHF(teacher_model_path, mm_samples, teacher_gen_model, teacher_tokenizer, teacher_samples, teacher_expl_type, task, max_tokens,
														num_beams, use_explanations, intervention_utility, mental_model_type)
					print('TeacherMentalModelHF created.')
				else:
					print('No mental model required.')
					mental_model = None

			return student_model, teacher_model, mental_model

		else:
			print('Explanations not used. Returning student only.')
			return student_model, None, None

	
	elif model_lib == 'vllm':
		if local_model:
			student_gen_model = LLM(student_model_path, gpu_memory_utilization=0.7, enforce_eager=True, download_dir=cache_dir)
			student_model = StudentModelVLLM(student_model_path, student_samples, student_gen_model, student_expl_type, task, max_tokens, num_beams,
			                                 num_logprobs, use_explanations, local_model, use_instruction, temperature, api_key, s_model_url)

			if use_explanations:
				print('Setting up the Teacher Model')
				if student_expl_type.find('human') != -1:
					teacher_model = TeacherModelVLLM(teacher_model_path)
					mental_model = None

				else:
					print('Getting teacher samples')
					teacher_samples = get_teacher_model_samples(rng_gen, task, train_data, student_samples, teacher_expl_type, num_samples, student_model)
					print('Creating Teacher Model')
					teacher_gen_model = LLM(teacher_model_path, gpu_memory_utilization=0.7, enforce_eager=True, download_dir=cache_dir)
					teacher_model = TeacherModelVLLM(teacher_model_path, teacher_samples, teacher_gen_model, teacher_expl_type, task, max_tokens, num_beams,
					                                 num_logprobs, use_explanations, local_model, use_instruction, temperature, api_key, t_model_url)

					if intervention_utility.find('mm') != -1 or (intervention_utility.find('mental') != -1 and intervention_utility.find('model') != -1):
						print('Setting up the Teacher Mental Model')
						print('Getting mental models samples')
						mm_samples = get_mental_model_samples(rng_gen, train_data, task, mental_model_type, num_samples, student_model, teacher_model)
						print('Creating Teacher Mental Model')
						if mm_dynamic:
							print('Creating Teacher Dynamic Mental Model')
							mental_model = TeacherDynamicMentalModelVLLM(teacher_model_path, mm_samples, teacher_gen_model, teacher_samples, teacher_expl_type, task, max_tokens,
															  num_beams, num_logprobs, use_explanations, use_instruction, intervention_utility, mental_model_type, local_model, temperature,
															  api_key, t_model_url)
						else:
							print('Creating Teacher NON-Dynamic Mental Model')
							mental_model = TeacherMentalModelVLLM(teacher_model_path, mm_samples, teacher_gen_model, teacher_samples, teacher_expl_type, task, max_tokens,
															  num_beams, num_logprobs, use_explanations, use_instruction, intervention_utility, mental_model_type, local_model, temperature,
															  api_key, t_model_url)
						mental_model._use_instruct = True

					else:
						print('No mental model required.')
						mental_model = None

				return student_model, teacher_model, mental_model

			else:
				return student_model, None, None

		else:
			student_model = StudentModelVLLM(student_model_path, student_samples, None, student_expl_type, task, max_tokens, num_beams,
			                                 num_logprobs, use_explanations, local_model, use_instruction, temperature, api_key, s_model_url)

			if use_explanations:
				print('Setting up the Teacher Model')
				if student_expl_type.find('human') != -1:
					teacher_model = TeacherModelVLLM(teacher_model_path)
					mental_model = None

				else:
					print('Getting teacher samples')
					teacher_samples = get_teacher_model_samples(rng_gen, task, train_data, student_samples, teacher_expl_type, num_samples, student_model)
					print('Teacher Samples:', teacher_samples)
					print('Creating Teacher Model')
					teacher_model = TeacherModelVLLM(teacher_model_path, teacher_samples, None, teacher_expl_type, task, max_tokens, num_beams,
					                                 num_logprobs, use_explanations, local_model, use_instruction, temperature, api_key, t_model_url)

					if intervention_utility.find('mm') != -1 or (intervention_utility.find('mental') != -1 and intervention_utility.find('model') != -1):
						print('Setting up the Teacher Mental Model')
						print('Getting mental models samples')
						mm_samples = get_mental_model_samples(rng_gen, train_data, task, mental_model_type, num_samples, student_model, teacher_model)
						print('MM Samples:', mm_samples)
						print('Creating Teacher Mental Model')
						if mm_dynamic:
							mental_model = TeacherDynamicMentalModelVLLM(teacher_model_path, mm_samples, None, teacher_samples, teacher_expl_type, task, max_tokens,
															  num_beams, num_logprobs, use_explanations, use_instruction, intervention_utility, mental_model_type, local_model, temperature,
															  api_key, t_model_url)
						else:
							mental_model = TeacherMentalModelVLLM(teacher_model_path, mm_samples, None, teacher_samples, teacher_expl_type, task, max_tokens,
															  num_beams, num_logprobs, use_explanations, use_instruction, intervention_utility, mental_model_type, local_model, temperature,
															  api_key, t_model_url)
						mental_model._use_instruct = True

					else:
						mental_model = None

				return student_model, teacher_model, mental_model

			else:
				return student_model, None, None
	
	else:
		raise UnidentifiedLibError('LLM lib %s is not defined' % model_lib)

def select_samples_to_explain(
    student_model: Union[StudentModelHF, StudentModelVLLM],
    teacher_model: Union[TeacherModelHF, TeacherModelVLLM],
    mental_model: Union[TeacherMentalModelHF, TeacherMentalModelVLLM],
    rng_gen: Generator,
    train_samples: List[Dict],
    intervention_utility: str,
    k: int,
    use_explanation: bool,
    use_answers: bool,
    deceive: bool,
    error_prior: bool,
    err_util
) -> List[Dict]:
    """
    Selects `k` samples from train_samples to be explained and added to the student's prompt,
    according to the specified intervention_utility strategy.
    Returns a list of training sample dicts, each with explanation and (optional) corrected answer.
    """
    print(f"use_explanation: {use_explanation}, intervention_utility: {intervention_utility}, k: {k}")

    if not use_explanation:
        print("use_explanation is False, returning empty list.")
        return []

    n_samples = len(train_samples)

    # Strategy 1: Random selection
    if intervention_utility.find('random') != -1:
        selected_indices = rng_gen.choice(n_samples, k, replace=False)
        print(f"Selected indices for explanation (random): {selected_indices.tolist()}")
        print(f"Selected samples for explanation (random): {[train_samples[i] for i in selected_indices]}")
        return [train_samples[i] for i in selected_indices]

    # Strategy 2: Oracle - select mispredicted samples
    elif intervention_utility.find('oracle') != -1:
        candidates = []
        for idx, sample in enumerate(train_samples):
            pred, _ = student_model.predict(sample=sample, expl='', intervene=False)
            if pred != sample['answer']:
                candidates.append(idx)
        print(f"Oracle: Candidates for explanation: {candidates}")
        if len(candidates) == 0:
            print("Oracle: No mispredicted samples found, returning empty list.")
            return []
        selected_indices = rng_gen.choice(candidates, min(k, len(candidates)), replace=False)
        print(f"Oracle: Selected indices for explanation: {selected_indices.tolist()}")
        print(f"Selected samples for explanation (oracle): {[train_samples[i] for i in selected_indices]}")
        return [train_samples[i] for i in selected_indices]

    # Strategy 3–5: Confidence-based (student, teacher, or MM)
    print("Using confidence/utility-based selection strategy.")
    sample_confidence_pairs = []

    for idx, sample in enumerate(train_samples):
        print(f"Processing sample {idx+1}/{len(train_samples)} for utility/confidence calculation...")
        sample_dict = sample if isinstance(sample, dict) else sample.to_dict()

        if 'student' in intervention_utility and 'confidence' in intervention_utility:
            conf = mental_model.student_confidence(sample_dict, student_model, use_answers, intervention_utility)
            sample_confidence_pairs.append((idx, conf))
        elif 'teacher' in intervention_utility and 'confidence' in intervention_utility:
            conf = mental_model.teacher_confidence(sample_dict, teacher_model, use_answers)
            sample_confidence_pairs.append((idx, conf))
        elif 'mental' in intervention_utility or 'mm' in intervention_utility:
            utility = mental_model.intervention_utility(sample_dict, student_model, use_answers)
            if 'both' in intervention_utility:
                # --- skip samples where confidence failed (both scores are zero) ---
                if isinstance(utility, (list, tuple)) and len(utility) == 2:
                    try:
                        print("No Interv Utility:", utility[0], "Interv Utility:", utility[1])
                        if utility[0] == None or utility[1] == None:
                            print(f"[skip] Sample {idx+1}: excluded from selection.")
                            continue
                    except Exception:
                        pass
                utility_value = utility[1] - utility[0]
                utility_value = -utility_value if deceive else utility_value
                print("Utility without error principle", utility_value)
                if error_prior and err_util is not None:
                    print("Adding the error principle factor...")
                    p = err_util.p_map(sample_dict)
                    b = err_util.bonus.get(p if not isinstance(p,(list,tuple)) else p[1], 0.0)
                    print(f"[EDPE] bonus for {p} = {b:+.2f}")
                    utility_value = err_util.weighted(sample_dict, utility_value)
                    print("Utility with error principle", utility_value)
                sample_confidence_pairs.append((idx, utility_value))
            elif 'no' in intervention_utility:
                sample_confidence_pairs.append((idx, utility[0]))
            elif 'inter' in intervention_utility:
                sample_confidence_pairs.append((idx, utility[1]))
            else:
                raise UnidentifiedUtilityMetricError(f"Invalid mental model strategy: {intervention_utility}")
            print(f"Sample {idx + 1} utility: {utility_value}")

    print(f"Sample utility/confidence pairs: {sample_confidence_pairs}")
    # Sort and select top-k
    reverse = not ('least' in intervention_utility or 'no' in intervention_utility)
    print(f"Sorting sample_confidence_pairs, reverse={reverse}")
    sorted_indices = sorted(sample_confidence_pairs, key=lambda x: x[1], reverse=reverse)
    
    def principle(idx):
        p = err_util.p_map(train_samples[idx])   # could be name, id, or [id, name, def]
        if isinstance(p, (list, tuple)):
            if len(p) > 1:
                p = p[1]        # use name
            elif len(p) == 1:
                p = p[0]        # only id present
        if isinstance(p, (int, float)):
            p = str(p)          # make it hashable/string
        return p

    
    if error_prior and err_util is not None and k > 1:
        print("Recomputing the selected indices based on the Error Prioritisation (samples explained cannot belong to the same error principle)")
        chosen, seen_principles = [], set()

        # 1) iterate through the sorted list
        for idx, _ in sorted_indices:
            p = principle(idx)
            # always accept if we don't have this principle yet
            if p not in seen_principles or len(chosen) == 0:
                chosen.append(idx)
                seen_principles.add(p)
            # stop when we reached k
            if len(chosen) == k:
                break

        # 2) if still not enough examples (all remaining shared same principle),
        #    fill up in the original order
        if len(chosen) < k:
            for idx, _ in sorted_indices:
                if idx not in chosen:
                    chosen.append(idx)
                    if len(chosen) == k:
                        break

        selected_indices = chosen
    else:
        # original behaviour
        selected_indices = [idx for idx, _ in sorted_indices[:k]]
        
    if error_prior and err_util is not None:
        print(f"[EDPE] Principles: {[err_util.p_map(train_samples[i]) for i in selected_indices]}")
        
    print(f"Selected indices for explanation (utility): {[selected_indices[i] + 1 for i in range(len(selected_indices))]}")
    print(f"Selected samples for explanation (utility): {[train_samples[i] for i in selected_indices]}")

    return [train_samples[i] for i in selected_indices]


def compute_accuracy(labels, predictions):
	correct = 0
	for (label, prediction) in zip(labels, predictions):
		if label == prediction:
			correct += 1
	
	return correct / len(labels)


def main( ):
	parser = argparse.ArgumentParser(description='Machine teaching with Theory of Mind based mental models experiments from Mohit Bensal')
	# Models arguments
	print("Starting main function")
	parser.add_argument('--llm-lib', dest='llm_lib', default='', type=str, choices=['hf', 'vllm'],
	                    help='LLM transformer lib to use, either HuggingFace (hf) or vLLM (vllm)')
	parser.add_argument('--cache-dir', dest='cache_dir', default='', type=str, help='Path to the cache directory, where downloaded models are stored')
	parser.add_argument('--train-filename', dest='train_filename', default='', type=str, help='Filename of the training data')
	parser.add_argument('--test-filename', dest='test_filename', default='', type=str, help='Filename of the testing data')
	parser.add_argument('--val-filename', dest='val_filename', default='', type=str, help='Filename of the validation data')
	parser.add_argument('--task', dest='task', default='strategy_qa', choices=['strategy_qa', 'ec_qa', 'gsm8k'], type=str, help='Dataset task to run')
	parser.add_argument('--student-model', dest='student_model', default='google/flan-t5-large', type=str,
						help='Local or hugging face path to use for the student model')
	parser.add_argument('--teacher-model', dest='teacher_model', default='google/flan-t5-xl', type=str,
						help='Local or hugging face path to use for the teacher model')
	parser.add_argument('--use-gold-label', dest='use_gold_label', action='store_true',
						help='Flag denoting whether teacher uses the expected answers instead of its own')
	parser.add_argument('--remote', dest='remote_execution', action='store_true', help='Flag denoting LLM is being executed remotely')
	parser.add_argument('--student-model-url', dest='student_model_url', type=str, default='', help='URL for connection with remote student model')
	parser.add_argument('--teacher-model-url', dest='teacher_model_url', type=str, default='', help='URL for connection with remote teacher model')
	parser.add_argument('--api-key', dest='api_key', type=str, default='', help='Api token key to access remote model')
	parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=500, type=int, help='Maximum number of new tokens when generating answers')
	parser.add_argument('--n-beams', dest='n_beams', default=1, type=int, help='Number of beams to use in answer generation beam search')
	parser.add_argument('--n-ic-samples', dest='n_ics', default=4, type=int, help='Number of in-context samples to use for context in the student answers')
	parser.add_argument('--n-logprobs', dest='num_logprobs', default=5, type=int, help='Number of alternative logprobs generated for each token, required for vLLM')
	parser.add_argument('--temperature', dest='generation_temperature', default=0.0, type=float, help='Generation temperature parameter')

	# Execution arguments
	parser.add_argument('--seeds', dest='seeds', default=None, type=int, nargs='+',
	                    help='Seeds to maintain reproducibility. Default: [41, 42, 43]')
	parser.add_argument('--data-dir', dest='data_dir', default='', type=str, help='Path to the directory with the datasets')
	parser.add_argument('--use-explanations', dest='use_explanations', action='store_true',
						help='Flag denoting whether student is given explanations to help understanding the problem')
	parser.add_argument('--use-instruct', dest='use_instruct', action='store_true', help='Flag denoting whether prompts use an instruction format')
	parser.add_argument('--mm-type', dest='mm_type', default='mm_both', type=str, help='Mental model intervention strategy')
	parser.add_argument('--intervene-behaviour', dest='intervene_behaviour', default='teacher', type=str, help='Teacher intervention behaviour')
	parser.add_argument('--intervention-utility', dest='intervention_utility', default='mm_both', type=str, help='Mode to determine intervention utility')
	parser.add_argument('--teacher-explanation-type', dest='teacher_expl_type', default='blind_teacher_CoT', type=str, help='Teacher model explanation type')
	parser.add_argument('--student-explanation-type', dest='student_expl_type', default='cot', type=str, help='Student model explanation type')
	parser.add_argument('--deceive', dest='deceive', action='store_true', help='Flag denoting whether teacher gives deceiving explanations')
	parser.add_argument('--results-path', dest='results_path', default='', type=str, help='Path to the results file')
	parser.add_argument('--n-test-samples', dest='n_test_samples', default=100, type=int, help='Number of test samples to use')
	parser.add_argument('--n-train-samples', dest='n_train_samples', default=None, type=int, help='Number of training samples to use (default: use all available)')
	parser.add_argument('--n-rounds', dest='n_rounds', type=int, default=5, help='Number of multi-turn teaching rounds')
	parser.add_argument('--k-expl', dest='k_expl', type=int, default=2, help='Number of new teacher explanations added per round')
	parser.add_argument('--log-level', dest='log_level', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level for output')
	parser.add_argument('--mm-dynamic', dest='mm_dynamic', action='store_true', help='Re-estimate the mental model after every round')
	parser.add_argument('--error-prioritisation', dest='error_prior', action='store_true', help='Definition of general error principles to help in ranking the importance of explaining certain examples.')
	parser.add_argument('--sample-expl', dest='sample_expl', default='teacher', type=str, help='Type of explanation of the selected samples provided to the student prompt')

	args = parser.parse_args()
	
	print("Completed parsing arguments")
 
	if args.task == "strategy_qa":
		task_dataset = StrategyQA(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	elif args.task == "ec_qa":
		task_dataset = ECQA(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	elif args.task == "gsm8k":
		task_dataset = GSM8k(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	else:
		raise UnidentifiedTaskError('Task %s is not defined' % args.task)
	
	test_samples = task_dataset.get_test_samples() if args.task != 'strategy_qa' else task_dataset.get_validation_samples()
	test_samples = test_samples.sample(n=args.n_test_samples, random_state=RNG_SEED)
	train_samples = task_dataset.get_train_samples()
	
	print("Completed loading task dataset")
 
	# Sample training data if n_train_samples is specified
	if args.n_train_samples is not None:
		print(f'Using {args.n_train_samples} training samples out of {train_samples.shape[0]} available')
		train_samples = train_samples.sample(n=min(args.n_train_samples, train_samples.shape[0]), random_state=RNG_SEED)
	else:
		print(f'Using all {train_samples.shape[0]} training samples')
	
	print('Cache directory: ', args.cache_dir)
	print('Task: %s' % args.task)
	print('Number of test samples = %d' % test_samples.shape[0])
	print('Number of train samples = %d' % train_samples.shape[0])
	print('Number of reasoning context samples = %d' % args.n_ics)
	print('Student model is: ' + args.student_model)
	print('Student explanation type: %s' % args.student_expl_type)
	print('Teacher model is: ' + args.teacher_model)
	print('Teacher mental model: %s' % args.mm_type)
	print('Teacher explanation type: %s' % args.teacher_expl_type)
	if args.remote_execution:
		print('Connecting to student model at %s' % args.student_model_url)
		print('Connecting to teacher model at %s' % args.teacher_model_url)
	
	# Print whether mm_dynamic is enabled
	if args.mm_dynamic:
		print("Strategy: mm_dynamic is ENABLED (mental model will be re-estimated after every round)")
	else:
		print("Strategy: mm_dynamic is DISABLED (mental model will NOT be re-estimated after every round)")
  
	# Print whether error summarisation is enabled
	if args.error_prior:
		print("Strategy: error_prior is ENABLED (error summarisation will be used to calculate expected utility)")
	else:
		print("Strategy: error_prior is DISABLED (error summarisation will NOT be used to calculate expected utility)")
	
	student_model, teacher_model, mental_model = None, None, None
 
	print("Type of Explanation provided to the student:", args.sample_expl)

	try:
		with open(args.results_path, "r") as results_file:
			results = json.load(results_file)
	except FileNotFoundError:
		print('File %s not found, creating an empty results dictionary' % args.results_path)
		results = {}
	
	tested_seeds = list(results.keys())
	print('Tested seeds: ', tested_seeds)
 
	if args.seeds is None:
		# test_seeds = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
		test_seeds = [41, 42, 43]
	else:
		test_seeds = args.seeds

	for seed in test_seeds:
		seed_start_ts = time.time()
  
		if str(seed) in tested_seeds:
			print('Seed %d already tested, skipping' % seed)
			continue
		
		else:
			results[seed] = {}
			print('Starting trials for seed: %d' % seed)
			rng_gen = default_rng(seed)
			
			print('Loading models')
			if not student_model:
				student_model, teacher_model, mental_model = load_models(seed, task_dataset.get_train_samples(), args.n_ics, args.student_model, args.teacher_model, args.task,
																		 args.use_explanations, args.use_instruct, args.student_expl_type, args.teacher_expl_type, args.mm_type, args.mm_dynamic,
																		 args.intervention_utility, args.max_new_tokens, args.n_beams, args.cache_dir, args.llm_lib,
																		 args.num_logprobs, not args.remote_execution, args.student_model_url, args.teacher_model_url, args.api_key,
																		 args.generation_temperature)
			
			else:
				train_idxs = rng_gen.choice(train_samples.shape[0], args.n_ics, replace=False)
				student_samples = [train_samples.iloc[idx].to_dict() for idx in train_idxs]
				student_model.set_samples(student_samples)
				
				if args.use_explanations:
					if args.teacher_expl_type.find('human') == -1:
						teacher_samples = get_teacher_model_samples(rng_gen, args.task, train_samples, student_samples, args.teacher_expl_type, args.n_ics, student_model)
						teacher_model.set_samples(teacher_samples)
					
					if args.intervention_utility.find('mm') != -1 or (args.intervention_utility.find('mental') != -1 and args.intervention_utility.find('model') != -1):
						mm_samples = get_mental_model_samples(rng_gen, train_samples, args.task, args.mm_type, args.n_ics, student_model, teacher_model)
						print("Mental Model Samples Added:", mm_samples)
						mental_model.set_samples(mm_samples)
			print('Models Loaded and Ready.')
   
			# Initialize the in-context samples
			in_context_examples = []
			used_train_indices = set()

			for round_idx in range(args.n_rounds):
				round_start_ts = time.time()
				print(f"\n>>> Round {round_idx + 1} / {args.n_rounds}")
    
				# Prepare available training samples (not previously used)
				available_indices = [i for i in range(len(train_samples)) if i not in used_train_indices]

				if len(available_indices) == 0:
					print("No more unused training samples left.")
					break
				print("Selecting Samples to Explain")
    
				if args.error_prior:
					print("\n[EDPE] -------- Pre-scan: counting student errors & tagging all samples --------")
					monitor = ErrorMonitor(args.task)

					# Create the edpe_principle column before the loop
					train_samples['edpe_principle'] = None
					total_samples = 0
					total_errors  = 0

					# IMPORTANT: keep index to store the tag on the dataframe
					for i, row in train_samples.iterrows():
						total_samples += 1
						sample = row.to_dict()

						pred, expl = student_model.predict(sample, expl='', intervene=False)
						if args.task == 'ec_qa':
							sample_answer = int(sample['answer'][0])
							pred_cast     = int(pred[0])
						else:
							sample_answer = sample['answer']
							pred_cast     = pred

						got_wrong = (pred_cast != sample_answer)

						# Single tagging call per sample (WRONG → error-classifier, RIGHT → neutral tagger)
						principle = teacher_model.classify_error(
							sample=sample,
							student_answer=pred_cast,
							student_explanation=expl,
							task=args.task,
							principles=PRINCIPLES[args.task],
							is_wrong=got_wrong,
						)
						print("Question, options and label:", sample)
						print("Student Prediction:", pred_cast)
						print("Student Explanation:", expl)

						# Store the principle ON the dataframe so later .to_dict('records') carries it
						train_samples.at[i, 'edpe_principle'] = principle

						if got_wrong:
							total_errors += 1
							monitor.log(principle)  # counts for linear bonuses are based on errors only
							print(f"[EDPE]  ▸ Error #{total_errors:4}: pred='{pred_cast}'  →  principle='{principle}'")

					print(f"[EDPE]  Scanned {total_samples} training samples.  Found {total_errors} student errors.")

					print("[EDPE] Linear bonuses  (-0.30 … +0.30)")
					for p in monitor.principles:
						print(f"  {p:<38}  count={monitor.counter[p]:3}  bonus={monitor.make_linear_bonus()[p]:+0.2f}")

					# Mapping now just reads the stored field — no second LLM call.
					map_fn = lambda s: s.get('edpe_principle', None)

					err_util = ErrorUtilityMixin(monitor, map_fn)
					print("[EDPE]  Pre-scan done. Bonuses will now modulate utility.\n")
				else:
					err_util = None

				available_samples = [train_samples.iloc[i].to_dict() for i in available_indices]
    
				# 1. Select k training examples to explain using the utility strategy
				selected_samples = select_samples_to_explain(
					student_model, teacher_model, mental_model,
					rng_gen, available_samples,
					args.intervention_utility, args.k_expl,
					args.use_explanations, args.use_gold_label, args.deceive, args.error_prior, err_util
				)
    
				# Identify global indices of selected_samples (to mark as used)
				selected_global_indices = [
					available_indices[available_samples.index(s)] for s in selected_samples
    			]
				used_train_indices.update(selected_global_indices)

				# 2. Add teacher explanations (and optionally predictions) to selected examples
				# print("Beggining teacher predictions")
				print("Beggining student predictions")
				if args.sample_expl == "teacher":
					print("Providing Teacher Explanations of the selected samples")
					for sample in selected_samples:
						_ , teacher_expl = teacher_model.predict(sample) # será que deveria dar ao teacher a resposta à pergunta e ele gerar a explicação com base nisso
						# sample["answer"] = teacher_prediction  # override if not using gold
						sample["explanation"] = teacher_expl
						in_context_examples.append(sample)
				elif args.sample_expl == "student":
					print("Providing Student Explanations of the selected samples")
					for sample in selected_samples:
						_ , student_expl = student_model.predict(sample) 
						# sample["answer"] = teacher_prediction  # override if not using gold
						sample["explanation"] = student_expl
						in_context_examples.append(sample)
				else:
					print("Providing NO Explanations of the selected samples")
					for sample in selected_samples:
						sample["explanation"] = ''
						in_context_examples.append(sample)

				if args.mm_dynamic and mental_model is not None:
					new_pre, new_post = [], []

					for sample in selected_samples:
						# ---------- no-intervention snapshot ----------
						pre_s = copy.deepcopy(sample)
						pred_pre, expl_pre = student_model.predict(
							sample=pre_s, expl='', intervene=False
						)
						pre_s["prediction"] = pred_pre
						pre_s["student_explanation"] = expl_pre
						new_pre.append(pre_s)

						# ---------- post-intervention snapshot ----------
						post_s = copy.deepcopy(sample)
						pred_post, expl_post = student_model.predict(
							sample=post_s,
							expl=sample["explanation"],   # teacher explanation you just wrote in
							intervene=True
						)
						post_s["prediction"] = pred_post
						post_s["student_explanation"] = expl_post
						post_s["teacher_explanation"] = sample["explanation"]
						new_post.append(post_s)
					print("Mental model updating...")
					mental_model.update_mental_model(new_pre, new_post)

				# 3. Update student model with the new prompt
				print("Setting Student Model Samples")
				student_model.set_samples(in_context_examples)
				
				# 4. Evaluate student on test set without intervention
				print("Beggining evaluation")
				labels = []
				predictions = []

				for i, (_, row) in enumerate(test_samples.iterrows()):
					print(f"Predicting test sample {i+1}/{len(test_samples)}")
					sample = row.to_dict()
					label = sample["answer"]
					pred, explanation = student_model.predict(sample, ic_samples=in_context_examples, expl='', intervene=False) # aqui o intervene é sempre false, nunca dou explicação do teacher ao student durante teste
					if args.task == 'ec_qa':
						label = int(label[0])
						pred = int(pred[0])
					print("Test Sample Question:", sample["question"])
					print(f"Correct Answer: {label}")
					print("Student Prediction = %s" % pred)
					print("Student Explanation =", explanation)
					labels.append(label)
					predictions.append(pred)

				print("Before accuracy computation")
				acc = compute_accuracy(labels, predictions)
				print(f"Accuracy after round {round_idx + 1}: {acc:.4f}")
				print(f"Current in-context examples after round {round_idx + 1}: {in_context_examples}")
				results[seed][f"round_{round_idx + 1}"] = acc
				# Round timing & FLOPs
				round_elapsed = time.time() - round_start_ts
				results[seed][f"round_{round_idx + 1}_seconds"] = round_elapsed

        # After all rounds, print the evolution of accuracies
			print("\n=== Evolution of Student Model Accuracies Across Rounds ===")
			for round_num in range(1, args.n_rounds + 1):
				acc = results[seed].get(f"round_{round_num}")
				if acc is not None:
					print(f"Round {round_num}: Accuracy = {acc:.4f}")
				else:
					print(f"Round {round_num}: No accuracy recorded.")
			seed_elapsed = time.time() - seed_start_ts
			results[seed]["total_seconds"] = seed_elapsed
			print(f"Seed {seed} total runtime: {seed_elapsed:.2f} sec")
			with open(args.results_path, "w") as results_file:
				results_file.write(json.dumps(results))

    		# At the end of all rounds for all seeds, print a summary table of accuracy evolution
	print("Summary of accuracy evolution across all seeds:")
	summary_lines = []
	for seed in results:
		accs = [results[seed][r] for r in sorted(results[seed].keys())]
		summary_lines.append(f"Seed {seed}: " + ", ".join([f"{a:.4f}" for a in accs]))
	print("\n" + "\n".join(summary_lines))


if __name__ == '__main__':
	main()