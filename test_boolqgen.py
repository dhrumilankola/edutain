from main import boolean_questions

payload = {
    "input_text": "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction. Particular applications of AI include expert systems, speech recognition, and machine vision.",
    "max_questions": 3
}

result = boolean_questions(payload)
print(result)
