#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Union, Tuple, Optional
from machine_teaching.models.vllm.model_vllm import ModelVLLM
from machine_teaching.models.model import UnidentifiedTaskError, UnidentifiedExplanationError
from vllm import SamplingParams
from pandas import DataFrame
from tqdm import tqdm
from openai import OpenAI
from utilities.prompts import TeachingPrompts


class StudentModel(ModelVLLM):
	
	def teacher_explanation_context(self, test_sample: Dict, teacher_explanation: str):
		prompt = TeachingPrompts.Student.teacher_explanation_prompt
		if self._task == "strategy_qa":
			examples = "\n\n".join([
					"Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			question = "\n\nQ: %s" % test_sample['question']
			explanation = "\nA: %s So the answer is " % teacher_explanation

		elif self._task == "ec_qa":
			examples = "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3], sample['options'][4], sample['explanation'], sample['answer'])
					 for sample in self._ic_samples])
			question = ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s" %
						(test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3], test_sample['options'][4]))
			explanation = "\nA: %s So the answer is " % teacher_explanation
		
		elif self._task == "gsm8k":
			examples = "\n\n".join(["Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			test_sample_explanation_sents = teacher_explanation.split(".")
			test_sample_partial_explanation = test_sample_explanation_sents[0] + "."
			question = "\n\nQ: %s" % test_sample['question']
			explanation = "\nA: %s So the answer is " % test_sample_partial_explanation
		
		else:
			raise UnidentifiedTaskError("Task %s not defined for teacher explanation context" % self._task)
		
		return prompt.format(context_examples=examples, sample_question=question, teacher_explanation=explanation)
	
	def get_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, ic_samples: List[Dict] = None):
		if not self._use_explanations:
			return self.no_explanation_context(sample, self._ic_samples)
		else:
			if intervene:
				return self.teacher_explanation_context(sample, explanation)
			elif self._explanation_type.find('cot') != -1 or (self._explanation_type.find('chain') != -1 and self._explanation_type.find('thought') != -1):
				return self.cot_context(sample, self._ic_samples)
			elif self._explanation_type.find('expl') != -1:
				return self.explanation_context(sample, self._ic_samples, explanation)
			elif self._explanation_type.find('rational') != -1:
				return self.rational_context(sample, self._ic_samples)
			elif self._explanation_type.find('no') != -1:
				return self.no_explanation_context(sample, self._ic_samples)
			else:
				raise UnidentifiedExplanationError("Explanation type '%s' not identified." % self._explanation_type)
	
	def predict_confidence(self, sample: Dict, with_explanation: bool = False, explanation: Union[List, str] = None, debug: bool = False) -> List[float]:
		# Get generation inputs
		prompt = self.get_context(sample, explanation=explanation)
		print("Prompt to predict confidence:", prompt)
		if self._use_instruct:
			if self._task == 'strategy_qa':
				system_prompt = TeachingPrompts.Student.student_system_prompt_instruct_strategy_qa
			elif self._task == 'ec_qa':
				system_prompt = TeachingPrompts.Student.student_system_prompt_instruct_ec_qa
			elif self._task == 'gsm8k':
				system_prompt = TeachingPrompts.Student.student_system_prompt_instruct_gsm8k
		else:
			system_prompt = TeachingPrompts.Student.student_system_prompt
		conversation = [
				{'role': 'system', 'content': system_prompt},
				{'role': 'user', 'content': prompt},
		]
		
		# Generate answer
		if self.local_model:
			gen_params = SamplingParams(
					temperature=self._temperature,
					top_k=self._num_beams,
					max_tokens=self._max_tokens,
					logprobs=self._n_logprobs
			)
			# outputs = self.gen_model.generate(context, gen_params)
			outputs = self.gen_model.chat(conversation, gen_params)
			
			# Get answer logprobs
			logprobs = outputs[0].outputs[0].logprobs
			tokens = outputs[0].outputs[0].token_ids
   
			answer_encode = self.gen_model.get_tokenizer().encode('ANSWER')[1]
			if answer_encode in tokens:
				answer_limits = [i for i, x in enumerate(tokens) if x == 'ANSWER']
				answer_limits = [answer_limits[0] + 2, answer_limits[1] - 1]
			else:
				nl_id = self.gen_model.get_tokenizer().encode('\n')[1]
				nldouble_id = self.gen_model.get_tokenizer().encode('\n\n')[1]
				nl_pos = tokens.index(nl_id) if nl_id in tokens else self._max_tokens
				nldouble_pos = tokens.index(nldouble_id) if nldouble_id in tokens else self._max_tokens
				answer_end = nl_pos if nl_pos < nldouble_pos else nldouble_pos
				answer_limits = [0, answer_end]
			
			tokens = tokens[answer_limits[0]:answer_limits[1]]
			logprobs = logprobs[answer_limits[0]:answer_limits[1]]
   
			print("=== Sliced tokens ===")
			print("Token IDs:", tokens)
			print("Decoded tokens:", self.gen_model.get_tokenizer().decode(tokens))

			if self._task == "strategy_qa":
				# Find model answer to question
				# no_id = self.gen_model.get_tokenizer().encode(' no')[1]
				# yes_id = self.gen_model.get_tokenizer().encode(' yes')[1]
    
				no_ids = self.gen_model.get_tokenizer().encode(' no', add_special_tokens=False)
				no_id = no_ids[0] if no_ids else -1
				yes_ids = self.gen_model.get_tokenizer().encode(' yes', add_special_tokens=False)
				yes_id = yes_ids[0] if yes_ids else -1
    
				print("Searching for token IDs: yes_id =", yes_id, "no_id =", no_id)
				print("yes_id in tokens?", yes_id in tokens)
				print("no_id in tokens?", no_id in tokens)

				no_pos = tokens.index(no_id) if no_id in tokens else self._max_tokens
				yes_pos = tokens.index(yes_id) if yes_id in tokens else self._max_tokens
				answer_pos = yes_pos if yes_pos < no_pos else no_pos

				# Get class scores
				if answer_pos < self._max_tokens:
					answer_logprobs = Tensor([logprob.logprob for logprob in logprobs[answer_pos].values()])
					answer_tokens_alt = list(logprobs[answer_pos].keys())
					scores = softmax(answer_logprobs, dim=-1)
					yes_score, no_score = scores[answer_tokens_alt.index(yes_id)], scores[answer_tokens_alt.index(no_id)]

				else:
					yes_score = 0.0
					no_score = 0.0

				class_scores = [yes_score, no_score]
				print('Yes score = %s' % yes_score)
				print('No score = %s' % no_score)

			elif self._task == "ec_qa":
				# Find model answer to question
				opt1_id = self.gen_model.get_tokenizer().encode('1')[1]
				opt2_id = self.gen_model.get_tokenizer().encode('2')[1]
				opt3_id = self.gen_model.get_tokenizer().encode('3')[1]
				opt4_id = self.gen_model.get_tokenizer().encode('4')[1]
				opt5_id = self.gen_model.get_tokenizer().encode('5')[1]
				opt1_pos = tokens.index(opt1_id) if opt1_id in tokens else self._max_tokens
				opt2_pos = tokens.index(opt2_id) if opt2_id in tokens else self._max_tokens
				opt3_pos = tokens.index(opt3_id) if opt3_id in tokens else self._max_tokens
				opt4_pos = tokens.index(opt4_id) if opt4_id in tokens else self._max_tokens
				opt5_pos = tokens.index(opt5_id) if opt5_id in tokens else self._max_tokens
				answer_pos = min([opt1_pos, opt2_pos, opt3_pos, opt4_pos, opt5_pos])

				# Get class scores
				if answer_pos < self._max_tokens:
					answer_logprobs = Tensor([logprob.logprob for logprob in logprobs[answer_pos].values()])
					answer_tokens_alt = list(logprobs[answer_pos].keys())
					scores = softmax(answer_logprobs, dim=-1)
					opt1_score = scores[answer_tokens_alt.index(opt1_id)]
					opt2_score = scores[answer_tokens_alt.index(opt2_id)]
					opt3_score = scores[answer_tokens_alt.index(opt3_id)]
					opt4_score = scores[answer_tokens_alt.index(opt4_id)]
					opt5_score = scores[answer_tokens_alt.index(opt5_id)]

				else:
					opt1_score = opt2_score = opt3_score = opt4_score = opt5_score = 0.0

				class_scores = [opt1_score, opt2_score, opt3_score, opt4_score, opt5_score]
				if debug:
					print('Option1 score = %s' % opt1_score)
					print('Option2 score = %s' % opt2_score)
					print('Option3 score = %s' % opt3_score)
					print('Option4 score = %s' % opt4_score)
					print('Option5 score = %s' % opt5_score)

			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)

		else:
			client = OpenAI(base_url=self.model_url, api_key=self.api_key)
			outputs = client.chat.completions.create(
					model=self.model_name,
					messages=conversation,
					max_completion_tokens=self._max_tokens,
					logprobs=True,
					top_logprobs=self._n_logprobs,
					temperature=self._temperature,
			)
    
			output_text = outputs.choices[0].message.content
			print("Raw output text for utility computation:", output_text)
			output_logprobs = outputs.choices[0].logprobs.content
			outputs_tokens = [elem.token for elem in output_logprobs]
			outputs_top_probs = [elem.top_logprobs for elem in output_logprobs]
			
			class_scores = self.get_prediction_confidence(output_text, outputs_tokens, outputs_top_probs, sample, debug)
			print("Class scores:", class_scores)
   
			client.close()

		return class_scores
	
	def predict(self, sample: Dict, ic_samples: List[Dict] = None, debug: bool = False, expl: Union[List, str] = None, intervene: bool = False):
		prompt = self.get_context(sample=sample, explanation=expl, intervene=intervene, ic_samples=ic_samples)

		if self._use_instruct:
			if self._task == 'strategy_qa':
				system_prompt = TeachingPrompts.Student.student_system_prompt_instruct_strategy_qa
			elif self._task == 'ec_qa':
				system_prompt = TeachingPrompts.Student.student_system_prompt_instruct_ec_qa
			elif self._task == 'gsm8k':
				system_prompt = TeachingPrompts.Student.student_system_prompt_instruct_gsm8k
		else:
			system_prompt = TeachingPrompts.Student.student_system_prompt
		conversation = [
				{'role': 'system', 'content': system_prompt},
				{'role': 'user', 'content': prompt},
		]
		if self.local_model:
			gen_params = SamplingParams(
					temperature=0.0,
					top_k=self._num_beams,
					max_tokens=self._max_tokens,
					logprobs=True,
					top_logprobs=self._n_logprobs
			)
			output_text = self.gen_model.chat(conversation, gen_params)[0].outputs[0].text

		else:
			client = OpenAI(base_url=self.model_url, api_key=self.api_key)
			outputs = client.chat.completions.create(
					model=self.model_name,
					messages=conversation,
					max_completion_tokens=self._max_tokens,
					logprobs=True,
					top_logprobs=self._n_logprobs,
					temperature=self._temperature,
			)
			output_text = outputs.choices[0].message.content.strip()
		
		prediction, explanation = self.get_prediction_from_output(output_text, sample, debug, prompt)

		if not self.local_model:
			client.close()

		return prediction, explanation
	
	def predict_batch(self, samples: DataFrame, intervention_indexes_per_budget: List[List[int]] = None, teacher: ModelVLLM = None, debug: bool = False) -> Tuple[List, List, List]:
		labels = []
		predictions_per_budget = [[] for _ in range(len(intervention_indexes_per_budget))]
		explanations_per_budget = [[] for _ in range(len(intervention_indexes_per_budget))]
		
		for test_index, test_sample in tqdm(samples.iterrows(), desc='Student Prediction Batch', total=samples.shape[0]):
			for i, intervention_indexes in enumerate(intervention_indexes_per_budget):
				if test_index in intervention_indexes:
					_, explanation = teacher.predict(sample=test_sample.to_dict())
					prediction_student, explanation_student = self.predict(sample=test_sample.to_dict(), expl=explanation, intervene=True, debug=debug)
				else:
					prediction_student, explanation_student = self.predict(sample=test_sample.to_dict(), expl='', intervene=False, debug=debug)
				predictions_per_budget[i].append(prediction_student)
				explanations_per_budget[i].append(explanation_student)
			labels.append(test_sample['answer'])
		
		return predictions_per_budget, explanations_per_budget, labels