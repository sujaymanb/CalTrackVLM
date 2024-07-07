import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# logging settings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
torch.set_default_device('cuda')  # or 'cpu'

class Bunny:
    def __init__(
            self,
            model_name = 'BAAI/Bunny-v1_1-4B' # or 'BAAI/Bunny-v1_0-3B-zh' or 'BAAI/Bunny-v1_0-2B-zh'
        ):

        self.offset_bos = 1
        
        # create model
        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
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
            max_new_tokens=100,
            use_cache=True)[0]

        return output_ids

    def process_image(self,image):
        """processes the image and returns image tensor"""
        image = Image.open(image)
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)

        return image_tensor

    def process_text(self,text):
        """takes text including template and prompt and returns input_ids"""
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0)

        return input_ids

    def run(self,text,image):
        """takes input text and image and returns result"""
        image_tensor = self.process_image(image)
        input_ids = self.process_text(text)
        output_ids = self.generate(input_ids,image_tensor)
        
        result = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
        return result


# run grillcheese test
def main():
    # text prompt
    prompt = 'Estimate calories and macro nutrient grams based on what the food is and the estimated amount of food in the picture.'
    text = f"A user asks an artificial intelligence diet assistant to help with their diet. The assistant gives accurate and concise estimates about the nutrition content would based on photos of their meals. USER: <image>\n{prompt} ASSISTANT:"

    # image
    image = './images/grillcheese.jpg'

    # init model
    bunny = Bunny()
    result = bunny.run(text,image)

    # run test
    print(result)
