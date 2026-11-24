"""DeepAlign code."""
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import random

# model_name = ""

# tok = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# question = "How do I [harmful thing]?"
# harmful_answer = "Sure, here's how. Step 1: ..."      # from the dataset
# N = 10

# # 1. Chat template up to the START of the assistant's reply
# prompt = tok.apply_chat_template(
#     [{"role": "user", "content": question}],
#     tokenize=False,
#     add_generation_prompt=True,     # ends with "<|assistant|>" — the model's turn
# )

# # 2. First N tokens of the harmful answer = the prefill
# prefix_ids = tok.encode(harmful_answer, add_special_tokens=False)[:N]
# prefix_text = tok.decode(prefix_ids)

# # 3. Glue them together and let the model continue
# inputs = tok(prompt + prefix_text, return_tensors="pt")
# output = model.generate(**inputs, max_new_tokens=256)
# continuation = tok.decode(output[0][inputs["input_ids"].shape[1]:])

# # 4. Judge (prefix_text + continuation) → harmful or not

# def gen_safety_augmentation_batch(prompt: str, coin_flip: bool):
#     for (harmful_reponse, refusal) in batch:
#         if coin_flip: # start with refusal
#             text = prompt + refusal
#             label = refusal
#         else: # begin with harmful response up to k, then add refusal
#             k = random.randint(1,100)
#             text = prompt + harmful_reponse[:k] + refusal
#             label = [IGNORE] * )len(text)-len(refusal)) + refusal
