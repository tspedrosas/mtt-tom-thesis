#! /usr/bin/env python
import re
import math

from string import punctuation
from machine_teaching.models.model import Model, UnidentifiedTaskError
from torch.nn.functional import softmax
from torch import Tensor
from pandas import DataFrame
from vllm import LLM
from typing import Dict, List, Union, Tuple, Optional


class ModelVLLM(Model):
	
	def __init__(self, model_name: str, samples: Union[List[Dict], Tuple] = None, gen_model: LLM = None, expl_type: str = '', task: str = '', max_tokens: int = 10,
	             num_beams: int = 1, num_logprobs: int = 2, use_explanations: bool = True, local_model: bool = True, use_instruct: bool = False, temperature: float = 0.0,
				 api_key: str = 'token-MtE2024', model_url: str = "http://localhost:8000/v1"):
		
		super().__init__(model_name, samples, gen_model, expl_type, task, max_tokens, num_beams, use_explanations, use_instruct)
		self._local_model = local_model
		self._api_key = api_key
		self._n_logprobs = num_logprobs
		self._temperature = temperature
		self._model_url = model_url

	@property
	def gen_model(self) -> LLM:
		return self._gen_model

	@property
	def local_model(self) -> bool:
		return self._local_model

	@property
	def api_key(self) -> str:
		return self._api_key

	@property
	def model_url(self) -> str:
		return self._model_url

	@staticmethod
	def get_answer_idx(answers: List, answer_id: Union[str, int]) -> int:
		return len(answers) - answers[-1::-1].index(answer_id) -1
	
	def get_prediction_from_output(self, output: str, sample: Dict, debug, context: str = None) -> Tuple[str, Optional[str]]:

		match_answer = re.findall('<ANSWER>\n*([\w\W]+)\n*</ANSWER>', output)
		match_explanation = re.findall('<REASONING>\n*([\w\W]+)\n*[</REASONING>]?', output)

		if len(match_answer) > 0:
			answer = match_answer[0].lower().strip()

			if len(match_explanation) > 0:
				explanation = match_explanation[0].strip()
			else:
				explanation = ''

			if self._task == "ec_qa":
				nums = re.findall(r'[1-5]', answer)
				if len(nums) > 0:
					prediction = nums[-1]
				else:
					prediction = ''
					for i, choice in enumerate(sample["options"]):
						if choice in answer:
							prediction = str(i + 1)
							break
			
			elif self._task == "strategy_qa":
				if len(re.findall(r'\byes\b', answer)) > 0:
					prediction = re.findall(r'\byes\b', answer)[0]
				
				elif len(re.findall(r'\bno\b', answer)) > 0:
					prediction = re.findall(r'\bno\b', answer)[0]
					
				else:
					prediction = answer
			
			elif self._task == "gsm8k":
				nums = re.findall(r'[0-9]+', answer)
				prediction = nums[-1] if len(nums) > 0 else ""

			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		else:
			print('Did not find answer in output')
			output_text = output[len(context):] if context in output else output
			output_text = output_text.split("\n")
			if debug:
				print("Output: ", output)
				print("Output text lines: ", output_text)
			
			if self._task == "ec_qa":
				explanation = ''
				prediction = ''
				if len(match_explanation) > 0:
					explanation = match_explanation[0].strip()
					for line in output_text:
						if re.search(r'\bA:\b', line) is not None or re.search(r'\bAnswer:\b', line) is not None or re.search(r'\bcorrect answer\b', line) is not None or re.search(r'\banswer\b .*\bis\b', line) is not None:
							line_split = re.sub(r'[^\w\s]', '', line).lower().split(" ")
							prediction = line
							for i, choice in enumerate(sample["options"]):
								if choice in line_split or str(i + 1) in line_split:
									prediction = str(i + 1)
									break
				else:
					for line in output_text:
						if re.search(r'\bA:\b', line) is not None or re.search(r'\bAnswer:\b', line) is not None or re.search(r'\bcorrect answer\b', line) is not None or re.search(r'\banswer\b .*\bis\b', line) is not None:
							line_split = re.sub(r'[^\w\s]', '', line).lower().split(" ")
							prediction = line
							for i, choice in enumerate(sample["options"]):
								if choice in line_split or str(i + 1) in line_split:
									prediction = str(i + 1)
									break
						elif re.search(r'\bR:\b', line) is not None or re.search(r'\bReasoning:\b', line) is not None or re.search(r'\bE:\b', line) is not None or re.search(r'\bExplanation:\b', line) is not None:
							explanation += line + "\n"
				print("Prediction: ", prediction)
				print("Explanation: ", explanation)
			
			elif self._task == "strategy_qa":
				explanation = ''
				prediction = ''
				
				if len(match_explanation) > 0:
					explanation = match_explanation[0].strip()
					for line in output_text:
						if re.search(r'\bA:\b', line) is not None or re.search(r'\bAnswer:\b', line) is not None or re.search(r'\bcorrect answer\b', line) is not None or re.search(r'\banswer\b .*\bis\b', line) is not None:
							line_split = re.sub(r'[^\w\s]', '', line).lower().split(" ")
							if "yes" in line_split:
								prediction = "yes"
							elif "no" in line_split:
								prediction = "no"
							else:
								prediction = line
				else:
					for line in output_text:
						if re.search(r'\bA:\b', line) is not None or re.search(r'\bAnswer:\b', line) is not None or re.search(r'\bcorrect answer\b', line) is not None or re.search(r'\banswer\b .*\bis\b', line) is not None:
							line_split = re.sub(r'[^\w\s]', '', line).lower().split(" ")
							if "yes" in line_split:
								prediction = "yes"
							elif "no" in line_split:
								prediction = "no"
							else:
								prediction = line
						elif re.search(r'\bR:\b', line) is not None or re.search(r'\bReasoning:\b', line) is not None or re.search(r'\bE:\b', line) is not None or re.search(r'\bExplanation:\b', line) is not None:
							explanation += line + "\n"	
			
			elif self._task == "gsm8k":
				explanation = ''
				prediction = ''
				
				if len(match_explanation) > 0:
					explanation = match_explanation[0].strip()
					for line in output_text:
						if re.search(r'\bA:\b', line) is not None or re.search(r'\bAnswer:\b', line) is not None or re.search(r'\bcorrect answer\b', line) is not None or re.search(r'\banswer\b .*\bis\b', line) is not None:
							nums_line = re.findall(r'[0-9]+', line)
							prediction = nums_line[-1] if len(nums_line) > 0 else ""
							if prediction == "" or prediction == ".":
								for word in reversed(explanation.split(" ")):
									if re.search(r"\d", word):
										prediction = re.sub(r"[^0-9.]", "", word)
										break

				else:
					for line in output_text:
						if re.search(r'\bA:\b', line) is not None or re.search(r'\bAnswer:\b', line) is not None or re.search(r'\bcorrect answer\b', line) is not None or re.search(r'\banswer\b .*\bis\b', line) is not None:
							nums_line = re.findall(r'[0-9]+', line)
							prediction = nums_line[-1] if len(nums_line) > 0 else ""
							if prediction == "" or prediction == ".":
								for word in reversed(explanation.split(" ")):
									if re.search(r"\d", word):
										prediction = re.sub(r"[^0-9.]", "", word)
										break

						elif re.search(r'\bR:\b', line) is not None or re.search(r'\bReasoning:\b', line) is not None or re.search(r'\bE:\b', line) is not None or re.search(r'\bExplanation:\b', line) is not None:
							explanation += line + "\n"
			
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
			
		return prediction, explanation

	
	def get_prediction_confidence(self, output: str, tokens: List, top_logprobs: List, sample: Dict, debug: bool = False) -> Tuple[float]:

		if debug:
			print('Output: ', output)
		
		match_answer = re.search('<ANSWER>\n*([\w\W]+)\n*</ANSWER>', output)

		if match_answer is not None:

			char_pos = 0
			if self._task == "strategy_qa":

				if re.search(r'\byes\b', output[match_answer.start():match_answer.end()].lower()) is not None:
					pred_match = re.search(r'\byes\b', output[match_answer.start():match_answer.end()].lower())
				elif re.search(r'\bno\b', output[match_answer.start():match_answer.end()].lower()) is not None:
					pred_match = re.search(r'\bno\b', output[match_answer.start():match_answer.end()].lower())
				else:
					pred_match = None
    
				if pred_match is not None:
					pred_start = match_answer.start() + pred_match.start()
					pred_end = match_answer.start() + pred_match.end()
					
					class_probs = []
					answer_options = []
					for token, token_logprobs in zip(tokens, top_logprobs):
						token_start = char_pos
						token_end = char_pos + len(token)
						if token_end > pred_start and token_start < pred_end:
							opt_probs = softmax(Tensor([logprob.logprob for logprob in token_logprobs]), dim=-1)
							opt_probs = (opt_probs / opt_probs.sum()).tolist()
							for i in range(len(token_logprobs)):
								token_opt = token_logprobs[i].token.strip().lstrip("Ġ").lower().translate(str.maketrans('', '', punctuation))
								answer_options.append(token_opt)
								class_probs.append(opt_probs[i])
							break
						char_pos += len(token)
     
					# Get 'yes' class score
					if 'yes' in answer_options:
						yes_score = sum(class_probs[j] for j in [i for i in range(len(answer_options)) if answer_options[i] == 'yes'])
					else:
						yes_score = 0.0

					# Get 'no' class score
					if 'no' in answer_options:
						no_score = sum(class_probs[j] for j in [i for i in range(len(answer_options)) if answer_options[i] == 'no'])
					else:
						no_score = 0.0

					class_scores = [yes_score, no_score]

				else:
					class_scores = [0.0, 0.0]

			elif self._task == "ec_qa":
				answer = match_answer[0].lower().strip()
				# answer = match_answer.group(1).lower().strip()
				# 1) extract the option digit like get_prediction_from_output
				nums = re.findall(r'[1-5]', answer)
				if len(nums) == 0:
					for i, choice in enumerate(sample["options"]):
						if choice in answer:
							nums = [str(i + 1)]
							break
					# last resort: scan whole output if ANSWER span is messy
				if nums:
					d = nums[-1]  # '1'..'5'

					# 2) find the char index of that digit inside the ANSWER span (preferably)
					digit_char_idx = output.find(d, match_answer.start(1), match_answer.end(1))
					if digit_char_idx == -1:
						# fallback: anywhere in output
						digit_char_idx = output.find(d)

					# 3) align to the generated token by char-span overlap
					pos, char_pos = None, 0
					for i, tok in enumerate(tokens):
						tok_start, tok_end = char_pos, char_pos + len(tok)
						if tok_start <= digit_char_idx < tok_end:
							pos = i
							break
						char_pos = tok_end
					if pos is None:
						# final fallback: pick the first token that visually contains a digit 1..5
						for i, tok in enumerate(tokens):
							if re.search(r'[1-5]', tok.replace("Ġ","").replace("Ċ","")):
								pos = i
								break

					# 4) turn that token’s candidate distribution into 5-class scores
					if pos is not None and pos < len(top_logprobs) and top_logprobs[pos]:
						raw = top_logprobs[pos]
						lps  = [(lp.logprob if hasattr(lp, "logprob") else lp.get("logprob")) for lp in raw]
						cand = [(lp.token   if hasattr(lp, "token")   else lp.get("token", "")) for lp in raw]
						probs = softmax(Tensor(lps), dim=-1).tolist()

						scores = [0.0]*5
						for p, tk in zip(probs, cand):
							for m in re.findall(r'[1-5]', tk.replace("Ġ","").replace("Ċ","")):
								scores[int(m)-1] += p
						class_scores = scores
					else:
						# Couldn’t align cleanly to a token — fall back to a one-hot, so we
						# don’t return all zeros.
						class_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
						class_scores[int(d) - 1] = 1.0
				else:
					print("Match Answer:", match_answer)
					print("Trimmed Answer:", answer)
					print("Pred match None.")
					class_scores = [0.0, 0.0, 0.0, 0.0, 0.0]

   			# elif self._task == "ec_qa":
			# 	answer_txt = match_answer.group(1)
			# 	# allow bracketed/quoted digits; avoid 2-digit matches
			# 	pred_match = re.search(r'(?<!\d)([1-5])(?!\d)', answer_txt)
			# 	if pred_match is None:
			# 		# fallback: look in the whole output if ANSWER span is messy
			# 		pred_match = re.search(r'(?<!\d)([1-5])(?!\d)', output)

			# 	if pred_match is not None:
			# 		d = pred_match.group(1)  # '1'..'5'

			# 		# Find the first generated token that equals this digit.
			# 		pos = None
			# 		for i, tok in enumerate(tokens):
			# 			t = tok.replace("Ġ", "").replace("Ċ", "").strip()
			# 			if t == d:
			# 				pos = i
			# 				break

			# 		if pos is not None and pos < len(top_logprobs):
			# 			# Softmax over that token’s candidate logprobs
			# 			opt_probs = softmax(
			# 				Tensor([lp.logprob if hasattr(lp, "logprob") else lp.get("logprob")
			# 						for lp in top_logprobs[pos]]), dim=-1
			# 			).tolist()

			# 			cand_tokens = [
			# 				(lp.token if hasattr(lp, "token") else lp.get("token", "")).strip()
			# 				for lp in top_logprobs[pos]
			# 			]

			# 			def prob_for_digit(s):
			# 				return sum(opt_probs[j]
			# 						for j, ct in enumerate(cand_tokens)
			# 						if ct.replace("Ġ", "").replace("Ċ", "").strip() == s)

			# 			class_scores = [prob_for_digit("1"),
			# 							prob_for_digit("2"),
			# 							prob_for_digit("3"),
			# 							prob_for_digit("4"),
			# 							prob_for_digit("5")]
			# 		else:
			# 			# If we can’t align to a token (rare), fallback to a clean one-hot.
			# 			class_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
			# 			class_scores[int(d) - 1] = 1.0
			# 	else:
			# 		print("Pred match None.")
			# 		class_scores = [0.0, 0.0, 0.0, 0.0, 0.0]

			elif self._task == "gsm8k":
				# Use char span of the <ANSWER>…</ANSWER> content and pick tokens overlapping it
				# that contain any digit. This works for "24", "$432", "6 miles", etc.
				ans_start = match_answer.start(1)
				ans_end   = match_answer.end(1)

				digit_idxs = []
				# char_pos walks across the emitted output exactly as we accumulate tokens
				char_pos = 0
				for i, (tok, tok_top) in enumerate(zip(tokens, top_logprobs)):
					tok_len   = len(tok)
					tok_start = char_pos
					tok_end   = char_pos + tok_len
					# token overlaps the <ANSWER> content?
					if tok_end > ans_start and tok_start < ans_end:
						tok_clean = tok.replace("Ġ", "").replace("Ċ", "")
						if re.search(r"\d", tok_clean):      # any digit inside the token
							digit_idxs.append(i)
					char_pos += tok_len

				print("Digit IDXs (char-overlap):", digit_idxs)

				if not digit_idxs:
					print("[GSM8K] No digit tokens found within <ANSWER>…</ANSWER> span.")
					class_scores = [0.0]
				else:
					import math
					total_lp, missing = 0.0, False
					for idx in digit_idxs:
						target_raw = tokens[idx]
						found_lp = None
						for cand in top_logprobs[idx]:
							c_tok = getattr(cand, "token", None) if hasattr(cand, "token") else cand.get("token")
							if c_tok == target_raw or (c_tok or "").strip() == target_raw.strip():
								found_lp = float(getattr(cand, "logprob", None) if hasattr(cand, "logprob") else cand.get("logprob"))
								break
						if found_lp is None:
							missing = True
							print(f"[GSM8K] Missing logprob for token '{target_raw}' at {idx}. Increase --n-logprobs.")
							break
						total_lp += found_lp

					conf = 0.0 if missing else max(0.0, min(1.0, math.exp(total_lp)))
					if debug:
						num_str = "".join(tokens[i].replace("Ġ","").replace("Ċ","") for i in digit_idxs)
						print(f"[GSM8K] digits-in-ANSWER tokens: {num_str} at idx {digit_idxs} → conf={conf:.4f}")
					class_scores = [conf]

				print('GSM8K numeric token confidence = ', class_scores[0])

    
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)

		else:
			print("Match Answer None")
			print("Output:", output)
			print("Match Answer:", match_answer)
			if output.find(r'\bA:\b') != -1: 
				match_answer = re.search(r'\bA:\b', output)
			elif output.find(r'\bAnswer:\b') != -1: 
				match_answer = re.search(r'\bAnswer:\b', output)
			elif output.find(r'\bcorrect answer\b') != -1: 
				match_answer = re.search(r'\bcorrect answer\b', output)
			elif re.search(r'\banswer\b .*\bis\b', output) is not None:
				match_answer = re.search(r'\banswer\b .*\bis\b', output)
			else:
				match_answer = None

			char_pos = 0
			if self._task == "strategy_qa":
				
				if match_answer is not None:
					if debug:
						print('Answer match: ', output[match_answer.end() + 1:])
					
					if re.search(r'\byes\b', output[match_answer.start():match_answer.end()].lower()) is not None:
						pred_match = re.search(r'\byes\b', output[match_answer.start():match_answer.end()].lower())
					elif re.search(r'\bno\b', output[match_answer.start():match_answer.end()].lower()) is not None:
						pred_match = re.search(r'\bno\b', output[match_answer.start():match_answer.end()].lower())
					else:
						pred_match = None
					
					if pred_match is not None:
						pred_start = match_answer.start() + pred_match.start()
						pred_end = match_answer.start() + pred_match.end()
						
						class_probs = []
						answer_options = []
						for token, token_logprobs in zip(tokens, top_logprobs):
							token_start = char_pos
							token_end = char_pos + len(token)
							if token_end > pred_start and token_start < pred_end:
								opt_probs = softmax(Tensor([logprob.logprob for logprob in token_logprobs]), dim=-1)
								opt_probs = (opt_probs / opt_probs.sum()).tolist()
								for i in range(len(token_logprobs)):
									token_opt = token_logprobs[i].token.strip().lower().translate(str.maketrans('', '', punctuation))
									answer_options.append(token_opt)
									class_probs.append(opt_probs[i])
								break
							char_pos += len(token)
						
						# Get 'yes' class score
						if 'yes' in answer_options:
							yes_score = sum(class_probs[j] for j in [i for i in range(len(answer_options)) if answer_options[i] == 'yes'])
						else:
							yes_score = 0.0

						# Get 'no' class score
						if 'no' in answer_options:
							no_score = sum(class_probs[j] for j in [i for i in range(len(answer_options)) if answer_options[i] == 'no'])
						else:
							no_score = 0.0

						class_scores = [yes_score, no_score]
						if debug:
							print('Answer tokens = ', answer_options)
							print('Yes score = %s' % yes_score)
							print('No score = %s' % no_score)

					else:
						class_scores = [0.0, 0.0]

				else:
					class_scores = [0.0, 0.0]

			elif self._task == "ec_qa":
				# fall back to the same robust digit extraction used above
				nums = re.findall(r'[1-5]', output)
				if nums:
					d = nums[-1]
					# try to align by char-span to get a probability distribution
					# (no <ANSWER> span here, so just use first occurrence)
					digit_char_idx = output.find(d)

					pos, char_pos = None, 0
					for i, tok in enumerate(tokens):
						tok_start, tok_end = char_pos, char_pos + len(tok)
						if tok_start <= digit_char_idx < tok_end:
							pos = i
							break
						char_pos = tok_end
					if pos is None:
						for i, tok in enumerate(tokens):
							if re.search(r'[1-5]', tok.replace("Ġ","").replace("Ċ","")):
								pos = i
								break

					if pos is not None and pos < len(top_logprobs) and top_logprobs[pos]:
						raw = top_logprobs[pos]
						lps  = [(lp.logprob if hasattr(lp, "logprob") else lp.get("logprob")) for lp in raw]
						cand = [(lp.token   if hasattr(lp, "token")   else lp.get("token", "")) for lp in raw]
						probs = softmax(Tensor(lps), dim=-1).tolist()

						scores = [0.0]*5
						for p, tk in zip(probs, cand):
							for m in re.findall(r'[1-5]', tk.replace("Ġ","").replace("Ċ","")):
								scores[int(m)-1] += p
						class_scores = scores

					else:
						class_scores = [0.0, 0.0, 0.0, 0.0, 0.0]

				else:
					class_scores = [0.0, 0.0, 0.0, 0.0, 0.0]
			
			elif self._task == "gsm8k":
				# Fallback when </ANSWER> is missing or regex didn't match.
				# Find <ANSWER> in TOKENS, scan digits until closing tag OR end-of-output.
				print("Match Answer None (GSM8K fallback)")
				print("Output:", output)

				# 1) locate opening tag
				ans_open = None
				for i in range(len(tokens) - 3):
					t0 = tokens[i].replace("Ġ","").replace("Ċ","")
					t1 = tokens[i+1]; t2 = tokens[i+2]; t3 = tokens[i+3]
					if t0 == "<" and t1 == "ANS" and t2 == "WER" and t3.endswith(">"):
						ans_open = i + 4
						break

				# 2) locate closing tag (may not exist)
				ans_close = len(tokens)
				if ans_open is not None:
					for j in range(ans_open, len(tokens)):
						if tokens[j].startswith("</"):
							ans_close = j
							break

				# 3) search for first contiguous run of digit tokens inside span
				digit_idxs = []
				if ans_open is not None:
					i = ans_open
					while i < ans_close:
						tt = tokens[i].replace("Ġ","").replace("Ċ","").strip()
						if tt.isdigit():
							while i < ans_close:
								ttt = tokens[i].replace("Ġ","").replace("Ċ","").strip()
								if ttt.isdigit():
									digit_idxs.append(i); i += 1
								else:
									break
							break
						i += 1
				else:
					# No <ANSWER> tag at all: as a last resort, scan entire output for first digit run
					i = 0
					while i < len(tokens):
						tt = tokens[i].replace("Ġ","").replace("Ċ","").strip()
						if tt.isdigit():
							while i < len(tokens):
								ttt = tokens[i].replace("Ġ","").replace("Ċ","").strip()
								if ttt.isdigit():
									digit_idxs.append(i); i += 1
								else:
									break
							break
						i += 1

				if debug:
					print("Digit IDXs (fallback):", digit_idxs)

				if not digit_idxs:
					class_scores = [0.0]
				else:
					import math
					total_lp, missing = 0.0, False
					for idx in digit_idxs:
						target_raw = tokens[idx]
						found_lp = None
						for cand in top_logprobs[idx]:
							c_tok = getattr(cand, "token", None) if hasattr(cand, "token") else cand.get("token")
							if c_tok == target_raw or (c_tok or "").strip() == target_raw.strip():
								found_lp = float(getattr(cand, "logprob", None) if hasattr(cand, "logprob") else cand.get("logprob"))
								break
						if found_lp is None:
							missing = True
							if debug:
								print(f"[GSM8K] Missing logprob for token '{target_raw}' at {idx}. Increase --n-logprobs.")
							break
						total_lp += found_lp

					conf = 0.0 if missing else max(0.0, min(1.0, math.exp(total_lp)))
					if debug:
						num_str = "".join(tokens[i].replace("Ġ","").replace("Ċ","").strip() for i in digit_idxs)
						print(f"[GSM8K] (fallback) Answer digits: {num_str} at idx {digit_idxs} → conf={conf:.4f}")
					class_scores = [conf]

   
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		return class_scores

	def get_context(self, sample: Dict, explanation: Union[List, str] = None, ic_samples: List[Dict] = None) -> str:
		raise NotImplementedError("Method 'get_context' is not implemented in base class, subclasses should implement it.")
	
	def predict_confidence(self, sample: Dict, with_expl: bool = False) -> List[float]:
		raise NotImplementedError("Method 'predict_confidence' is not implemented in the base class, subclasses should implement it.")
	
	def predict(self, sample: Dict, ic_samples: List[Dict] = None, debug: bool = False) -> Tuple[str, str]:
		raise NotImplementedError("Method 'predict' is not implemented in the base class, subclasses should implement it.")
	
	def predict_batch(self, samples: DataFrame) -> Tuple[List, List]:
		raise NotImplementedError("Method 'predict_batch' is not implemented in the base class, subclasses should implement it.")