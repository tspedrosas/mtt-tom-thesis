#! /usr/bin/env python

from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Union, Tuple
from machine_teaching.models.vllm.teacher_model_vllm import TeacherModel
from machine_teaching.models.vllm.student_model_vllm import StudentModel
from vllm import LLM, SamplingParams
from machine_teaching.models.model import UnidentifiedTaskError, UnidentifiedUtilityMetricError
from openai import OpenAI
from utilities.prompts import TeachingPrompts


class TeacherMentalModel(TeacherModel):
	
	def __init__(self, model_name: Union[str, List[str]], intervention_samples: Union[List[Dict], Tuple] = None, gen_model: Union[LLM, List[LLM]] = None,
	             teacher_samples: List[Dict] = None, expl_type: str = '', task: str = '', max_tokens: int = 10, num_beams: int = 1, num_logprobs: int = 2,
	             use_explanations: bool = True, use_instruct: bool = False, utility_type: str = '', mm_type: str = 'mm_both', local_model: bool = False, temperature: float = 0.0,
	             api_key: str = None, model_url: str = "http://127.0.0.1:8000/v1"):
		
		super().__init__(model_name, intervention_samples, gen_model, expl_type, task, max_tokens, num_beams, num_logprobs, use_explanations, local_model, use_instruct,
		                 temperature, api_key, model_url)
		
		self._teacher_samples = teacher_samples.copy() if teacher_samples is not None else None
		self._mm_type = mm_type
		self._utility_type = utility_type
	
	@property
	def teacher_samples(self) -> List[Dict]:
		return self._teacher_samples
	
	@property
	def mm_type(self) -> str:
		return self._mm_type
	
	@property
	def utility_type(self) -> str:
		return self._utility_type
	
	def get_student_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, use_answers: bool = False, debug: bool = False) -> str:
		raise NotImplementedError("Method 'get_student_context' is not implemented in teacher mental model base class, subclasses should implement it.")
	
	def predict_prompt(self, prompt: str, test_sample: Dict, debug: bool = False) -> Tuple:
		if self._use_instruct:
			if self._task == "strategy_qa":
				system_prompt = TeachingPrompts.Teacher.mm_teacher_instruct_strategy_qa_system_prompt
			elif self._task == "ec_qa":
				system_prompt = TeachingPrompts.Teacher.mm_teacher_instruct_ec_qa_system_prompt
			else:
				system_prompt = TeachingPrompts.Teacher.mm_teacher_instruct_system_prompt
		else:
			system_prompt = TeachingPrompts.Teacher.mm_teacher_system_prompt
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
			generated = self.gen_model.chat(conversation, gen_params)
			
			# Get answer logprobs
			prediction = generated[0].outputs[0].text
			logprobs = generated[0].outputs[0].logprobs
			tokens = generated[0].outputs[0].token_ids
   
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

			# Get the answer in text
			prediction = prediction[len(prompt):] if prompt in prediction else prediction
			prediction = prediction[:prediction.index('\n')].strip() if '\n' in prediction else prediction.strip()
			if debug:
				print('Mental model prediction = ', prediction)

			if self._task == "strategy_qa":
				print('Recognized task Strategy QA')
				# Find model answer to decode
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
				opt1_encoded = self.gen_model.get_tokenizer().encode('1')
				opt1_id = opt1_encoded[1] if len(opt1_encoded) > 1 else opt1_encoded[0]
				opt2_encoded = self.gen_model.get_tokenizer().encode('2')
				opt2_id = opt2_encoded[1] if len(opt2_encoded) > 1 else opt2_encoded[0]
				opt3_encoded = self.gen_model.get_tokenizer().encode('3')
				opt3_id = opt3_encoded[1] if len(opt3_encoded) > 1 else opt3_encoded[0]
				opt4_encoded = self.gen_model.get_tokenizer().encode('4')
				opt4_id = opt4_encoded[1] if len(opt4_encoded) > 1 else opt4_encoded[0]
				opt5_encoded = self.gen_model.get_tokenizer().encode('5')
				opt5_id = opt5_encoded[1] if len(opt5_encoded) > 1 else opt5_encoded[0]
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
			prediction, _ = self.get_prediction_from_output(output_text, test_sample, debug, prompt)
			if debug:
				print('Mental model prediction = ', prediction)

			outputs_tokens = [elem.token for elem in output_logprobs]
			outputs_top_probs = [elem.top_logprobs for elem in output_logprobs]
			class_scores = self.get_prediction_confidence(output_text, outputs_tokens, outputs_top_probs, test_sample, debug)

			client.close()

		return class_scores, prediction

	def simulate_utility(self, sample: Dict, use_answers: bool, debug: bool = False) -> Tuple:
		
		if use_answers:
			correct_answer = sample["answer"]
		else:
			print("Using teacher prediction")
			teacher_prediction, _ = self.predict(sample, debug=debug)
			correct_answer = teacher_prediction
		
		if self._mm_type.find('both') != -1:
			no_inter_context = self.get_student_context(sample, None, False, use_answers)
			no_inter_scores, no_inter_output = self.predict_prompt(no_inter_context, sample, debug=debug)
			
			inter_context = self.get_student_context(sample, None, True, use_answers)
			inter_scores, inter_output = self.predict_prompt(inter_context, sample, debug=debug)
			
			print('No intervention output: %s\t No intervention scores = ' % no_inter_output, no_inter_scores)
			print('Intervention output: %s\t Intervention scores = ' % inter_output, inter_scores)

			if self._task == "strategy_qa":
				if no_inter_scores == [0.0, 0.0] and inter_scores == [0.0, 0.0]:
					return [no_inter_output, inter_output], [None, None]
				elif no_inter_scores == [0.0, 0.0]:
					no_inter_scores = [None, None]
				elif inter_scores == [0.0, 0.0]:
					inter_scores = [None, None]
    
				if correct_answer == "yes":
					return [no_inter_output, inter_output], [no_inter_scores[0], inter_scores[0]]
				else:
					return [no_inter_output, inter_output], [no_inter_scores[1], inter_scores[1]]
			
			elif self._task == "ec_qa":
				if no_inter_scores == [0.0, 0.0, 0.0, 0.0, 0.0] and inter_scores == [0.0, 0.0, 0.0, 0.0, 0.0]:
					return [no_inter_output, inter_output], [None, None]
				elif no_inter_scores == [0.0, 0.0, 0.0, 0.0, 0.0]:
					no_inter_scores = [None, None]
				elif inter_scores == [0.0, 0.0, 0.0, 0.0, 0.0]:
					inter_scores = [None, None]
     
				# Add safety checks to prevent IndexError
				if len(no_inter_scores) < 5 or len(inter_scores) < 5:
					print(f"Warning: Scores list too short. no_inter_scores length: {len(no_inter_scores)}, inter_scores length: {len(inter_scores)}")
					return [no_inter_output, inter_output], [None, None]
				
				if correct_answer == "1":
					return [no_inter_output, inter_output], [no_inter_scores[0], inter_scores[0]]
				elif correct_answer == "2":
					return [no_inter_output, inter_output], [no_inter_scores[1], inter_scores[1]]
				elif correct_answer == "3":
					return [no_inter_output, inter_output], [no_inter_scores[2], inter_scores[2]]
				elif correct_answer == "4":
					return [no_inter_output, inter_output], [no_inter_scores[3], inter_scores[3]]
				else:
					return [no_inter_output, inter_output], [no_inter_scores[4], inter_scores[4]]
			
			elif self._task == "gsm8k":
				return [no_inter_output, inter_output], [no_inter_scores[0], inter_scores[0]]
			
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		else:
			if self._mm_type.find('no') != -1:
				context = self.get_student_context(sample, None, False, use_answers)
			else:
				context = self.get_student_context(sample, None, True, use_answers)
			option_scores, output = self.predict_prompt(context, sample, debug=debug)
			
			if self._task == "strategy_qa":
				if correct_answer == "yes":
					return output, option_scores[0]
				else:
					return output, option_scores[1]
			
			elif self._task == "ec_qa":
				if correct_answer == "1":
					return output, option_scores[0]
				elif correct_answer == "2":
					return output, option_scores[1]
				elif correct_answer == "3":
					return output, option_scores[2]
				elif correct_answer == "4":
					return output, option_scores[3]
				else:
					return output, option_scores[4]
			
			elif self._task == "gsm8k":
				return output, option_scores[0]
			
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
	
	def intervention_utility(self, sample: Dict, student: StudentModel, use_answers: bool, debug: bool = False) -> Union[float, Tuple]:
		
		if self._utility_type.find('student') != -1 and self._utility_type.find('confidence') != -1:
			if self._utility_type.find('intervention') != -1:
				if self._utility_type.find('no') != -1:
					class_scores = student.predict_confidence(sample, debug=debug)
				else:
					_, explanation = self.predict(sample, ic_samples=self.teacher_samples, debug=debug)
					class_scores = student.predict_confidence(sample, with_explanation=True, explanation=explanation, debug=debug)
				if self._task == 'strategy_qa':
					if sample["answer"] == "yes":
						return class_scores[0]
					else:
						return class_scores[1]
				elif self._task == 'ec_qa':
					return class_scores[int(sample["answer"]) - 1]
				elif self._task == 'gsm8k':
					pass
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			elif self._utility_type.find('least') != -1:
				class_scores = student.predict_confidence(sample, debug=debug)
				return min(class_scores)
			elif self._utility_type.find('utility') != -1 and self._utility_type.find('correct') != -1:
				_, explanation = self.predict(sample, ic_samples=self.teacher_samples, debug=debug)
				intervention_class_scores = student.predict_confidence(sample, with_explanation=True, explanation=explanation, debug=debug)
				no_intervention_class_scores = student.predict_confidence(sample, with_explanation=False, explanation='', debug=debug)
				if self._task == 'strategy_qa':
					if sample["answer"] == "yes":
						return intervention_class_scores[0] - no_intervention_class_scores[0]
					else:
						return intervention_class_scores[1] - no_intervention_class_scores[1]
				elif self._task == 'ec_qa':
					return intervention_class_scores[int(sample["answer"]) - 1] - no_intervention_class_scores[int(sample["answer"]) - 1]
				elif self._task == 'gsm8k':
					pass
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			else:
				raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)
		
		elif self._utility_type.find('teacher') != -1 and self._utility_type.find('confidence') != -1:
			class_scores = self.predict_confidence(sample, with_explanation=True, ic_samples=self.teacher_samples, debug=debug)
			if self._task == "strategyQA":
				return class_scores[0] if sample["answer"] == "yes" else class_scores[1]
			elif self._task == "ecqa":
				return class_scores[int(sample["answer"]) - 1]
			elif self._task == 'gsm8k':
				pass
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		elif (self._utility_type.find('mental') != -1 and self._utility_type.find('model') != -1) or self._utility_type.find('mm') != -1:
			if debug:
				print('Simulating utility for sample: ', sample['question'])
			_, scores = self.simulate_utility(sample, use_answers, debug=debug)
			return scores
		
		else:
			raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)