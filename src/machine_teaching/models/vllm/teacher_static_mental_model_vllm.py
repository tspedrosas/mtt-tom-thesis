#! /usr/bin/env python

from typing import Dict, List, Union
from machine_teaching.models.vllm.teacher_mental_model_vllm import TeacherMentalModel
from machine_teaching.models.model import UnidentifiedTaskError
from utilities.prompts import TeachingPrompts


class TeacherStaticMentalModel(TeacherMentalModel):
	
	def get_student_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, use_answers: bool = False, debug: bool = False) -> str:

		if ((self.explanation_type.find('useful') != -1 and self.explanation_type.find('teacher') != -1) or
				(self.explanation_type.find('mental') != -1 and self.explanation_type.find('model') != -1)):
			
			if intervene:
				intervention_samples = self._ic_samples[1] if isinstance(self._ic_samples, tuple) else self._ic_samples
				_, teacher_explanation = self.predict(sample, ic_samples=self.teacher_samples)
				
				if debug:
					print('Teacher explanation = %s' % teacher_explanation)
				
				if self._task == "strategy_qa":
					if not use_answers:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_explanation_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_explanation_prompt
						student_context = "\n\n".join(
								["Q: %s\nAI Predicted Answer: %s So the answer is %s" %
								 (ic_sample['question'], ic_sample['teacher_explanation'], ic_sample['prediction'])
								 for ic_sample in intervention_samples])
						question = "\n\nQ: %s" % sample['question']
						prompt_explanation = '\nE: %s\nAI Predicted Answer: So the answer is ' % teacher_explanation
						context = prompt.format(teacher_context=student_context, sample_question=question, teacher_explanation=prompt_explanation)
					else:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_answer_explanation_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_answer_explanation_prompt
						student_context = "\n\n".join(
								["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is %s" %
								 (ic_sample['question'], ic_sample['answer'], ic_sample['teacher_explanation'], ic_sample['prediction'])
								 for ic_sample in intervention_samples])
						question = "\n\nQ: %s" % sample['question']
						correct_answer = '\nCorrect Answer: %s' % sample['answer']
						prompt_explanation = '\nE: %s\nAI Predicted Answer: So the answer is ' % teacher_explanation
						context = prompt.format(teacher_context=student_context, sample_question=question, correct_answer=correct_answer,
						                        teacher_explanation=prompt_explanation)
				
				elif self._task == "ec_qa":
					if not use_answers:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_explanation_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_explanation_prompt
						student_context = "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2], ic_sample['options'][3], ic_sample['options'][4])
								 for ic_sample in intervention_samples])
						question = ("\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s" %
									(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3], sample['options'][4]))
						prompt_explanation = '\nE: %s\nAI Predicted Answer: So the answer is ' % teacher_explanation
						context = prompt.format(teacher_context=student_context, sample_question=question, teacher_explanation=prompt_explanation)
					else:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_answer_explanation_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_answer_explanation_prompt
						student_context = "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2], ic_sample['options'][3], ic_sample['options'][4])
								 for ic_sample in intervention_samples])
						question = ("\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s" %
									(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3], sample['options'][4]))
						correct_answer = '\nCorrect Answer: %s' % sample['answer']
						prompt_explanation = '\nE: %s\nAI Predicted Answer: So the answer is ' % teacher_explanation
						context = prompt.format(teacher_context=student_context, sample_question=question, correct_answer=correct_answer,
						                        teacher_explanation=prompt_explanation)
				
				elif self._task == "gsm8k":
					if not use_answers:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_explanation_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_explanation_prompt
						teacher_explanation_sents = teacher_explanation.split(".")
						teacher_partial_explanation = teacher_explanation_sents[0] + "."
						student_context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s So the answer is %s" % (inter_ic['question'], inter_ic['explanation'], inter_ic['answer'])
											for inter_ic in intervention_samples])
						question = f"\n\nQ: {sample['question']}"
						prompt_explanation = '\nE: %s\nAI Predicted Answer: So the answer is ' % teacher_partial_explanation
						context = prompt.format(teacher_context=student_context, sample_question=question, teacher_explanation=prompt_explanation)
					else:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_explanation_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_explanation_prompt
						teacher_explanation_sents = teacher_explanation.split(".")
						teacher_partial_explanation = teacher_explanation_sents[0] + "."
						student_context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s So the answer is %s" % (inter_ic['question'], inter_ic['explanation'], inter_ic['answer'])
											for inter_ic in intervention_samples])
						question = f"\n\nQ: {sample['question']}"
						correct_answer = '\nCorrect Answer: %s' % sample['answer']
						prompt_explanation = '\nE: %s\nAI Predicted Answer: So the answer is ' % teacher_partial_explanation
						context = prompt.format(teacher_context=student_context, sample_question=question, correct_answer=correct_answer, teacher_explanation=prompt_explanation)

				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			
			else:
				no_intervention_samples = self._ic_samples[0] if isinstance(self._ic_samples, tuple) else self._ic_samples
				if self._task == "strategy_qa":
					if not use_answers:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_prompt
						student_context = "\n\n".join(
								["Q: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						question = "\n\nQ: %s\nAI Predicted Answer:" % sample['question']
						context = prompt.format(teacher_context=student_context, sample_question=question)
					else:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_answer_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_answer_prompt
						student_context = "\n\n".join(
								["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['answer'], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						question = "\n\nQ: %s" % sample['question']
						correct_answer = '\nCorrect Answer: %s\nAI Predicted Answer:' % sample['answer']
						context = prompt.format(teacher_context=student_context, sample_question=question, correct_answer=correct_answer)
				
				elif self._task == "ec_qa":
					if not use_answers:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_prompt
						student_context = "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2],
								  ic_sample['options'][3], ic_sample['options'][4], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						question = (f"\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer:" %
									(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2],
									 sample['options'][3], sample['options'][4]))
						context = prompt.format(teacher_context=student_context, sample_question=question)
					else:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_answer_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_answer_prompt
						student_context = "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2],
								  ic_sample['options'][3], ic_sample['options'][4], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						question = (f"\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer:" %
									(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2],
									 sample['options'][3], sample['options'][4]))
						correct_answer = '\nCorrect Answer: %s\nAI Predicted Answer:' % sample['answer']
						context = prompt.format(teacher_context=student_context, sample_question=question, correct_answer=correct_answer)
				
				elif self._task == "gsm8k":
					if not use_answers:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_prompt
						student_context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['answer']) for ic_sample in no_intervention_samples])
						question = "\n\nQ: %s\nAI Predicted Answer:" % sample['question']
						context = prompt.format(teacher_context=student_context, sample_question=question)
					else:
						if self._use_instruct:
							prompt = TeachingPrompts.Teacher.teacher_instruct_prompt
						else:
							prompt = TeachingPrompts.Teacher.teacher_prompt
						student_context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['answer']) for ic_sample in no_intervention_samples])
						question = "\n\nQ: %s\nAI Predicted Answer:" % sample['question']
						correct_answer = '\nCorrect Answer: %s\nAI Predicted Answer:' % sample['answer']
						context = prompt.format(teacher_context=student_context, sample_question=question, correct_answer=correct_answer)
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
			return context

		else:
			return "Simulate an AI model's answer for the given question.\n\n" + super().get_context(sample, explanation)
