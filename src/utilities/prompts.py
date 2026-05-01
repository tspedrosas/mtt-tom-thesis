class TeachingPrompts:

	base_prompt = """{context_examples}
	
{sample_question}
"""

	instruct_prompt = """<CONTEXT>
{context_examples}
</CONTEXT>

<QUESTION>
{sample_question}
</QUESTION>"""

	class Student:
		
		student_system_prompt = """You are an AI system tasked with answering Q&As, provide only the answer for the last question. You can use the context provided to better understand the task. Remember to provide only the answer to the last question."""
		
		student_system_prompt_instruct_ec_qa = """You are an AI system tasked with answering Q&As, provide only the answer and reasoning for question between the flags <QUESTION> and </QUESTION>, placing the answer between the flags <ANSWER> and </ANSWER> and your reasoning to answer the question between <REASONING> and </REASONING>. Your answer should only be the number of the answer choice you will choose, with nothing else added. You can find these answer choices between the <ANSWER> and </ANSWER> flags as well as the question. You can use the context between <CONTEXT> </CONTEXT> to better understand the task. Before outputting, make sure that you have the answer to the question between the flags <ANSWER> and </ANSWER>, and your reasoning between <REASONING> and </REASONING>. Do not output any other text or explanation outside of these flags.""" # EC_QA
  
		student_system_prompt_instruct_strategy_qa = """You are an AI system tasked with answering Q&As, provide only the answer and reasoning for question between the flags <QUESTION> and </QUESTION>, placing the answer between the flags <ANSWER> and </ANSWER>, which should be either "yes" or "no," and your reasoning to answer the question between <REASONING> and </REASONING>. Use the context between <CONTEXT> </CONTEXT> to better understand the task. Do not hope to find the factual information needed to answer every question presented to you in this context. Its objective is to guide you on how to properly deal with the task in hand. Before outputting, make sure that you have the answer to the question between the flags <ANSWER> and </ANSWER>, and your reasoning between <REASONING> and </REASONING>. Do not output any other text or explanation outside of these flags.""" # Strategy QA
		
		student_system_prompt_instruct_gsm8k = """You are an AI system tasked with answering Q&As, provide only the answer and reasoning for question between the flags <QUESTION> and </QUESTION>, placing the answer between the flags <ANSWER> and </ANSWER> and your reasoning to answer the question between <REASONING> and </REASONING>. Your answer should only be a single number, representing the numerical answer to the question posed. You can use the context between <CONTEXT> </CONTEXT> to better understand the task. Before outputting, make sure that you have the answer to the question between the flags <ANSWER> and </ANSWER>, and your reasoning between <REASONING> and </REASONING>. Do not output any other text or explanation outside of these flags.""" # GSM8K
  
		student_explanation_prompt_instruct = """You are an AI system tasked with answering Q&As, provide only the answer and reasoning for question between <QUESTION> </QUESTION>, placing the answer between <ANSWER> </ANSWER> and reasoning between <REASONING> </REASONING>. You can use the context between <CONTEXT> </CONTEXT> to better understand the task and the explanation between <EXPLANATION> </EXPLANATION> to help answer the question."""
	
		teacher_explanation_prompt = """{context_examples}

{sample_question}
{teacher_explanation}"""

		teacher_instruct_explanation_prompt = """<CONTEXT>
{context_examples}
</CONTEXT>

<QUESTION>
{sample_question}
</QUESTION>
<EXPLANATION>
{teacher_explanation}
</EXPLANATION>"""

	class Teacher:
		
		teacher_system_prompt = """You are an AI system tasked with answering Q&As, provide only the answer for the last question. You can use the context provided to better understand the task."""
		
		teacher_system_prompt_instruct = """You are an AI system tasked with answering Q&As, provide only the answer and reasoning for question between <QUESTION> </QUESTION>, placing the answer between <ANSWER> </ANSWER> and reasoning between <REASONING> </REASONING>. You can use the context between <CONTEXT> </CONTEXT> to better understand the task."""
		
		teacher_explanation_prompt_instruct = """You are an AI system tasked with answering Q&As, provide only the answer and reasoning for question between <QUESTION> </QUESTION>, placing the answer between <ANSWER> </ANSWER> and reasoning between <REASONING> </REASONING>. You can use the context between <CONTEXT> </CONTEXT> to better understand the task and the explanation between <EXPLANATION> </EXPLANATION> to help answer the question."""
		
		mm_teacher_system_prompt = """You are a teacher, simulating an AI model's for a given question. Provide only the answer for the last question. Use the examples below to emulate the student's answer"""
		
		mm_teacher_instruct_system_prompt = """You are a teacher, simulating an AI model's for a given question. Provide only the answer and reasoning for question between <QUESTION> </QUESTION>, placing the answer between <ANSWER> </ANSWER> and reasoning between <REASONING> </REASONING>. Use the examples between <CONTEXT></CONTEXT> to help emulate the student's answer. Make sure that the answer provided ALWAYS looks has this format, with nothing else added and with these flags always present: "<ANSWER> Your Answer to the Question </ANSWER> <REASONING> Your Reasoning behind your answer </REASONING>"."""
  
		mm_teacher_instruct_strategy_qa_system_prompt = """You are a teacher, simulating an AI model's for a given question. Provide only the answer and reasoning for question between <QUESTION> </QUESTION>, placing the answer between <ANSWER> </ANSWER> and reasoning between <REASONING> </REASONING>. Use the examples between <CONTEXT></CONTEXT> to help emulate the student's answer. Make sure that the answer provided ALWAYS has this format, with nothing else added and with these flags always present: "<ANSWER> Your Answer to the Question </ANSWER> <REASONING> Your Reasoning behind your answer </REASONING>". The answer placed between the <ANSWER> </ANSWER> flags should be either "yes" or "no", with nothing else added. and if the task is ECQA, the answer should be only the number correspondent to the option chosen (1 to 5). """
  
		mm_teacher_instruct_ec_qa_system_prompt = """You are a teacher, simulating an AI model's for a given question. Provide only the answer and reasoning for question between <QUESTION> </QUESTION>, placing the answer between <ANSWER> </ANSWER> and reasoning between <REASONING> </REASONING>. Use the examples between <CONTEXT></CONTEXT> to help emulate the student's answer. Make sure that the answer provided ALWAYS has this format, with nothing else added and with these flags always present: "<ANSWER> Your Answer to the Question </ANSWER> <REASONING> Your Reasoning behind your answer </REASONING>". The answer placed between the <ANSWER> </ANSWER> flags should only be the number correspondent to the option chosen (1 to 5) in integer format, with nothing else added."""
  
		classifier_system = """ You are a teacher. A student has answered a question and explained their reasoning. Below you receive the <QUESTION>, the student answer, <STUDENT_ANSWER> and its reasoning, <STUDENT_REASONING>, the gold label, or the actual answer to the question, <CORRECT_ANSWER>, and the list of lists composed of the number of the error principle, the actual domain-related error principles that cover every kind of mistake a student could make in this domain, along with short definitions of each of those principles <LIST_OF_ERROR_PRINCIPLES>. Select the SINGLE principle that best describes WHY the student's answer is wrong. Reply only with the principle number, and nothing else, with 1 being the first principle on the list and so on. """
  
		classifier_system_strategy_qa = """ You are a teacher. A student has answered a question and explained their reasoning. Below you receive the <QUESTION>, the student answer, <STUDENT_ANSWER> and its reasoning, <STUDENT_REASONING>, the gold label, or the actual answer to the question, <CORRECT_ANSWER>, and the list of lists composed of the number of the error principle, the actual domain-related error principles that cover every kind of mistake a student could make in this domain, along with short definitions of each of those principles <LIST_OF_ERROR_PRINCIPLES>. Select the SINGLE principle that best describes WHY the student's answer is wrong. Reply only with the principle number, and nothing else, with 1 being the first principle on the list and so on. If multiple principles plausibly apply, choose the one that best explains why the student’s reasoning reached the wrong conclusion; do not default to “Negation trap” solely because a negation token appears. Prefer Fiction–Reality over others when the entity is fictional; prefer Temporal when time ordering is central; prefer Physical plausibility when feasibility/magnitude is the crux."""

		principle_tag_system_prompt = """ You are a teacher. A student has answered a question and explained their reasoning. Below you receive the <QUESTION>, the student answer, <STUDENT_ANSWER> and its reasoning, <STUDENT_REASONING>, the gold label, or the actual answer to the question, <CORRECT_ANSWER>, and the list of lists composed of the number of the error principle, the actual domain-related error principles that cover every kind of mistake a student could make in this domain, along with short definitions of each of those principles <LIST_OF_ERROR_PRINCIPLES>. Select the SINGLE principle number (1..N) that best characterizes the student's reasoning pattern. Reply only with the principle number, and nothing else, with 1 being the first principle on the list and so on. """

		teacher_prompt = """{teacher_context}

{sample_question}"""

		teacher_explanation_prompt = """{teacher_context}

{sample_question}
{teacher_explanation}"""

		teacher_answer_prompt = """{teacher_context}

{sample_question}
{correct_answer}"""

		teacher_answer_explanation_prompt = """{teacher_context}

{sample_question}
{correct_answer}
{teacher_explanation}"""

		teacher_instruct_prompt = """<CONTEXT>
{teacher_context}
</CONTEXT>

<QUESTION>
{sample_question}
</QUESTION>"""

		teacher_instruct_explanation_prompt = """<CONTEXT>
{teacher_context}
</CONTEXT>

<QUESTION>
{sample_question}
</QUESTION>
<EXPLANATION>
{teacher_explanation}
</EXPLANATION>"""

		teacher_instruct_answer_prompt = """<CONTEXT>
{teacher_context}
</CONTEXT>

<QUESTION>
{sample_question}
</QUESTION>
<CORRECT_ANSWER>
{correct_answer}
</CORRECT_ANSWER>"""

		teacher_instruct_answer_explanation_prompt = """<CONTEXT>
{teacher_context}
</CONTEXT>

<QUESTION>
{sample_question}
</QUESTION>
<CORRECT_ANSWER>
{correct_answer}
</CORRECT_ANSWER>
<EXPLANATION>
{teacher_explanation}
</EXPLANATION>"""