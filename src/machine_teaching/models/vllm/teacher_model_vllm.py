#! /usr/bin/env python
import re

from pandas import DataFrame
from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Tuple, Union
from machine_teaching.models.vllm.model_vllm import ModelVLLM
from machine_teaching.models.model import UnidentifiedTaskError, UnidentifiedExplanationError
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI
from utilities.prompts import TeachingPrompts


class TeacherModel(ModelVLLM):
	
	def __init__(self, model_name: str, samples: List[Dict] = None, gen_model: LLM = None, expl_type: str = '', task: str = '', max_tokens: int = 10, num_beams: int = 1,
	             num_logprobs: int = 2, use_explanations: bool = True, local_model: bool = True, use_instruct: bool = False, temperature: float = 0.0, api_key: str = 'token-MtE2024',
	             model_url: str = "http://localhost:8000/v1"):
		
		super().__init__(model_name, samples, gen_model, expl_type, task, max_tokens, num_beams, num_logprobs, use_explanations, local_model, use_instruct, temperature, api_key, model_url)
	
	def get_context(self, sample: Dict, explanation: Union[List, str] = None, ic_samples: List[Dict] = None) -> str:
		if ic_samples is None:
			ic_samples = self._ic_samples[0] if isinstance(self._ic_samples, tuple) else self._ic_samples

		if not self._use_explanations:
			return self.no_explanation_context(sample, ic_samples)
		else:
			if self._explanation_type.find('blind') != -1:
				if self._explanation_type.find('rational') != -1:
					return self.rational_context(sample, ic_samples)
				elif self._explanation_type.find('cot') != -1 or (self._explanation_type.find('chain') != -1 and self._explanation_type.find('thought') != -1):
					return self.cot_context(sample, ic_samples)
			elif self._explanation_type.find('useful') != -1:
				return self.cot_context(sample, ic_samples)
			elif self._explanation_type.find('expl') != -1:
				return self.explanation_context(sample, ic_samples, explanation)
			else:
				raise UnidentifiedExplanationError("Explanation type '%s' not identified." % self._explanation_type)
	
	def teacher_system_prompt(self, sample: Dict, with_explanation: bool = False, debug: bool = False, ic_samples: List[Dict] = None) -> List[float]:
		# Get generation inputs
		prompt = self.get_context(sample, explanation='', ic_samples=ic_samples)
		if self._use_instruct:
			system_prompt = TeachingPrompts.Teacher.teacher_system_prompt_instruct
		else:
			system_prompt = TeachingPrompts.Teacher.teacher_system_prompt
		conversation = [
				{'role': 'system', 'content': system_prompt},
				{'role': 'user', 'content': prompt},
		]

		if self.local_model:
			gen_params = SamplingParams(
					temperature=0.0,
					top_k=self._num_beams,
					max_tokens=self._max_tokens,
					logprobs=self._n_logprobs
			)

			# Generate answer
			outputs = self.gen_model.chat(conversation, gen_params)

			# Get answer logprobs
			logprobs = outputs[0].outputs[0].logprobs
			tokens = outputs[0].outputs[0].token_ids
   
			print("Raw output text:", outputs[0].outputs[0].text)
			print("All token IDs:", tokens)
			print("All decoded tokens:", self.gen_model.get_tokenizer().decode(tokens))
   
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
				if debug:
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
			output_logprobs = outputs.choices[0].logprobs.content

			outputs_tokens = [elem.token for elem in output_logprobs]
			outputs_top_probs = [elem.top_logprobs for elem in output_logprobs]
   
			print("--------------WITHOUT SELF MODEL--------------")
			print("TM - Raw output text:", output_text)
			print("TM - All token IDs:", outputs_tokens)
			print("TM - Top Probs:", outputs_top_probs)
   
			class_scores = self.get_prediction_confidence(output_text, outputs_tokens, outputs_top_probs, sample, debug)
   
			print("TM - Class scores:", class_scores)

			client.close()

		return class_scores
	
	def predict(self, sample: Dict, ic_samples: List[Dict] = None, debug: bool = False) -> Tuple[str, str]:
		if self._explanation_type.find("human") != -1:
			return str(sample["answer"]), str(sample["explanation"])
		
		else:
			prompt = self.get_context(sample, explanation='', ic_samples=ic_samples)
			if self._use_instruct:
				system_prompt = TeachingPrompts.Teacher.teacher_system_prompt_instruct
			else:
				system_prompt = TeachingPrompts.Teacher.teacher_system_prompt
			conversation = [
					{'role': 'system', 'content': system_prompt},
					{'role': 'user', 'content': prompt},
			]

			if self.local_model:
				gen_params = SamplingParams(
						temperature=0.0,
						top_k=self._num_beams,
						max_tokens=self._max_tokens,
						logprobs=self._n_logprobs
				)
				output = self.gen_model.chat(conversation, gen_params)[0].outputs[0].text

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
				output = outputs.choices[0].message.content.strip()

			prediction, explanation = self.get_prediction_from_output(output, sample, debug, prompt)

			if not self.local_model:
				client.close()

			return prediction, explanation
	
	def predict_batch(self, samples: DataFrame, debug: bool = False) -> Tuple[List, List]:
		predictions = []
		explanations = []
		
		for test_index, test_sample in tqdm(samples.iterrows(), desc='Teacher Prediction Batch', total=samples.shape[0]):
			prediction, explanation = self.predict(sample=test_sample.to_dict(), debug=debug)
			predictions.append(prediction)
			explanations.append(explanation)
		
		return predictions, explanations

	def _build_principle_prompt(self,
								question: str,
								student_answer: str,
								student_expl: str,
								correct_answer: str,
								task: str,
								principles: list[str],
        						is_wrong: bool) -> list[dict]:
		"""Return a chat-format conversation ready for self.gen_model.chat()."""
		numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(principles))
		prompt = (
			f"<QUESTION>:\n{question}\n\n"
			f"<STUDENT_ANSWER>:\n{student_answer}\n\n"
			f"<STUDENT_REASONING>:\n{student_expl or 'N/A'}\n\n"
			f"<CORRECT_ANSWER>:\n{correct_answer}\n\n"
			f"<LIST_OF_ERROR_PRINCIPLES>:\n{numbered}\n\n"
			"Answer with the principle number only."
		)
		if is_wrong:
			system_prompt = (TeachingPrompts.Teacher.classifier_system_strategy_qa
							if task == 'strategy_qa' else
							TeachingPrompts.Teacher.classifier_system)
		else:
			system_prompt = TeachingPrompts.Teacher.principle_tag_system_prompt
		return [
			{"role": "system", "content": system_prompt},
			{"role": "user",   "content": prompt},
		]

	def classify_error(self,
					sample: dict,
					student_answer: str,
					student_explanation: str,
					task: str,
					principles: list[str],
     				is_wrong: bool) -> str:
		"""
		Return the *string* principle selected by the teacher LLM
		for this (wrong) student answer.
		"""
		conversation = self._build_principle_prompt(
			question=sample["question"],
			student_answer=student_answer,
			student_expl=student_explanation,
			correct_answer=sample["answer"],
			task = task,
			principles=principles,
			is_wrong=is_wrong,  
		)

		if self.local_model:
			gen_params = SamplingParams(
				temperature=0.0,
				top_k=self._num_beams,
				max_tokens=4,
				logprobs=0,
			)
			raw = self.gen_model.chat(conversation, gen_params)[0].outputs[0].text
		else:
			client = OpenAI(base_url=self.model_url, api_key=self.api_key)
			resp = client.chat.completions.create(
				model=self.model_name,
				messages=conversation,
				max_completion_tokens=4,
				temperature=0.0,
				logprobs=False,
			)
			raw = resp.choices[0].message.content.strip()
			client.close()

		try:
			idx = int(raw) - 1
			return principles[idx]
		except (ValueError, IndexError):
			return principles[-1]  # default “Other” bucket
