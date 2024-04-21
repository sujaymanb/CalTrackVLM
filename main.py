from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import transformers
from PIL import Image
import warnings
from langchain_core.prompts import PromptTemplate

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
torch.set_default_device('cpu')  # or 'cuda'

model_name = 'BAAI/Bunny-v1_0-3B' # or 'BAAI/Bunny-v1_0-3B-zh' or 'BAAI/Bunny-v1_0-2B-zh'
# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

# load huggingface pipeline for langchain
pipe = pipeline("vqa", model=model, tokenizer=tokenizer)

hf = HuggingFacePipeline(pipeline=pipe)

# create chain
template = """ A user asks an artificial intelligence diet assistant to help with their diet. The assistant gives accurate and concise estimates about the nutrition content would based on photos of their meals. USER: <image>\n{question} ASSISTANT: Here's the nutrition information for your meal:"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "Estimate calories and macro nutrient grams based on what the food is and the estimated amount of food in the picture."

image = Image.open('./images/grillcheese.jpg')
image_tensor = model.process_images([image],model.config).to(dtype=model.dtype)

print(chain.invoke({"question": question, "image": image_tensors}))

