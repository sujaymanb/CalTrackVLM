import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# logging settings
#transformers.logging.set_verbosity_error()
#transformers.logging.disable_progress_bar()
#warnings.filterwarnings('ignore')

class Bunny:
    def __init__(
            self,
            model_name = 'BAAI/Bunny-v1_1-4B',
            device = 'cpu'
        ):

        # set device
        self.device = device
        torch.set_default_device(device)  # or 'cpu'

        self.offset_bos = 1
        
        # create model
        # if self.device == 'cpu':
        #     dtype = torch.float32
        # else:
        #     dtype = torch.float16

        dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=dtype,
                            device_map='auto',
                            trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            trust_remote_code=True)


    def generate(self,input_ids,image_tensor):
        """takes input ids and image tensor to generate output ids"""
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=500,
            use_cache=True,
            repetition_penalty=1.0)[0]

        return output_ids

    def process_image(self,filename):
        """processes the image and returns image tensor"""
        image = Image.open(filename)
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype, device=self.device)

        return image_tensor

    def process_text(self,text):
        """takes text including template and prompt and returns input_ids"""
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][self.offset_bos:], dtype=torch.long).unsqueeze(0).to(self.device)

        return input_ids

    def run(self,text,image):
        """takes input text and image and returns result"""
        image_tensor = self.process_image(image)
        input_ids = self.process_text(text)
        output_ids = self.generate(input_ids,image_tensor)
        
        result = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return result



'''grillcheese test'''
def main():
    # text prompt
    #prompt = 'Estimate calories and macro nutrient grams based on what the food is and the estimated amount of food in the picture.'
    prompt = """Given the picture of the meal above, identify individual food items in the image. 
For each item in the picture, give the following pieces of information in a form that is parseable in python as a list of dictionaries, with each item being a dictionary containing the following keys:
a. name
b. weight
b. calories
c. protein
d. fat
e. carbs
f. fiber
g. sugar

Output only the data structure and nothing else so that it is parseable with python's ast.literal_eval().
"""

    text = f"You are a diet tracking assistant. USER: <image>\n{prompt} OUTPUT:"

    # image
    image = './images/grillcheese.jpg'

    # init model
    bunny = Bunny()
    result = bunny.run(text,image)

    # run test
    print(result)

'''run test'''
main()