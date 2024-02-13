#TIPS:
# Use delimiters <>, ``` ```
# Prompt the model to ouput stuff in JSON format
# few shot prompting
# specify steps to complete task,
# instruct model to work out its own solution before conclusion
# (own idea: maybe try to prompt the model to find what part of the context is relevant
# to the claim, if no part is to answer False. Then to output that part and given that part, to
# reach a conclusion )

prompt_format_original = """
Answer the question about {topic} based on the given context.\n\n

{context}

Input: {claim} Is this statement True or False? Answer using a single word in [\"True\", \"False\"].
Output:
"""


prompt_format1 = """
You are given a claim and a reference. You have to answer whether the claim is "True" or "False" based on the reference.
The reference is between <> and the claim is between ``` ```. 
In case the reference does not contain information about the claim, the answer should be "False".

Here are some examples:
< The sky is blue. >
Is the following claim faithful to the given reference? 
``` The sky is red. ```
Answer "True" or "False":
<ANSWER> False <ANSWER>

< The sky is red. >
Is the following claim faithful to the given reference? 
``` The sky is red. ```
Answer "True" or "False":
<ANSWER> True <ANSWER>

Given the following reference about {topic}:
< {reference} >
Is the following claim faithful to the given reference? 
``` {claim} ```
Answer "True" or "False":
"""

prompt_format2 = """
You are given a claim and a reference. You have to answer whether the claim is "True" or "False" based on the reference.
The reference is between <> and the claim is between ``` ```.

To solve this task, first find the relevant part of the reference that supports the claim. 
Based on this relevant part, reach a conclusion.
In case the reference does not contain information about the claim, the answer should be "False".

Here are some examples:

Given the following reference about the weather:
<  There are 30 degrees celsius outside. The sky is red. It is raining >
Is the following claim faithful to the given reference? 
``` The sky is red. ```
Answer "True" or "False":
<ANSWER> True <ANSWER>

Given the following reference about the weather:
< There are 30 degrees celsius outside. The sky is red. It is raining. >
Is the following claim faithful to the given reference? 
``` The sky is blue. ```
Answer "True" or "False":
<ANSWER> False <ANSWER>

Given the following reference about the weather:
< There are 30 degrees celsius outside. It is raining. >
Is the following claim faithful to the given reference? 
``` The sky is blue. ```
Answer "True" or "False":
<ANSWER> False <ANSWER>


Given the following reference about {topic}:
< {reference} >
Is the following claim faithful to the given reference? 
``` {claim} ```
Answer "True" or "False":
"""


prompt_format3 = """
You are given a claim and a reference. You have to answer whether the claim is {{True}} or {{False}}" based on the reference.
The reference is between <> and the claim is between ``` ```.

To solve this task, first find the relevant part of the reference that supports the claim. 
Based on this relevant part, reach a conclusion.
In case the reference does not contain information about the claim, the answer should be {{False}}".

Here are some examples:

Given the following reference about the weather:
<  There are 30 degrees celsius outside. The sky is red. It is raining >
Is the following claim faithful to the given reference? 
``` The sky is red. ```
Answer {{True}} or {{False}}":
{{True}}

Given the following reference about the weather:
< There are 30 degrees celsius outside. The sky is red. It is raining. >
Is the following claim faithful to the given reference? 
``` The sky is blue. ```
Answer {{True}} or {{False}}":
{{False}}

Given the following reference about the weather:
< There are 30 degrees celsius outside. It is raining. >
Is the following claim faithful to the given reference? 
``` The sky is blue. ```
Answer {{True}} or {{False}}":
{{False}}


Given the following reference about {topic}:
< {reference} >
Is the following claim faithful to the given reference? 
``` {claim} ```
Answer {{True}} or {{False}}":
"""

def get_prompt(topic, context, fact):
    prompt = prompt_format3.format(topic=topic, reference=context, claim=fact)
    return prompt
