import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from util import KeywordStoppingCriteria
from PIL import Image
import warnings
from conversation import conv_template

# disable warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set pytorch device
torch.set_default_device('cuda')  # or 'cpu'

# constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

class BunnyHelper:
    def __init__(self,
            model_path: str = 'BAAI/Bunny-v1_0-3B',
            device_map: str = 'cuda',
            torch_dtype,
            temperature,
            max_new_tokens
            ) -> None:
        self.model = None
        self.tokenizer = None
        self.img_tensor = None
        self.conv = None
        self.conv_img = None
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.load_models(model_path,
                        device_map=device_map)

    def load_models(self, model_path: str,
                    device_map: str,
                    torch_dtype = torch.float16) -> None:
        """ Load the model and tokenizer """
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True)


    def tokenizer_image_token(prompt):
        prompt_chunks = [self.tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])

        return input_ids


    def process_image(self, img_path: str) -> None:
        # load image
        self.conv_img = Image.open(img_path).convert('RGB')

        # process image
        img_tensor = self.model.process_image([self.conv_img],self.model.config)to(dtype=self.model.dtype)

        if type(img_tensor) is list:
            self.img_tensor = [image.to(model.device, dtype=model.dtype) for image in img_tensor]
        else:
            self.img_tensor = img_tensor.to(model.device, dtype=model.dtype)


    def generate_answer(self, debug=False) -> str:
        """ Generate an answer from the current conversation """
        raw_prompt = self.conv.get_prompt()
        input_ids = self.tokenizer_image_token(raw_prompt).unsqueeze(0).to(self.model.device)

        # stop keywords
        keywords = [self.conv.end]
        stopping_criteria = KeywordStoppingCriteria(keywords)

        # stream text
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # do inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=self.img_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # process outputs
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs

        return output.split


    def get_conv_text(self) -> str:
        """ return full conversation text """
        return self.conv.get_prompt()


    def start_chat(self,
                img_path: str,
                prompt: str):
        # setup new convo
        self.conv = conv_template.copy()

        # load and process image
        self.process_image(img_path)

        # append to convo history
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        self.conv.append_message("USER", prompt)
        conv.append_message("ASSISTANT", None)

        return self.generate_answer()

    def continue_chat(self,
                    prompt: str):
        """ chain/continue existing conversation """
        if self.conv is None:
            raise RuntimeError("No existing conversation text.")

        self.conv.append_message("USER", prompt)
        self.conv.append_message("ASSISTANT", None)
        
        return self.generate_answer()


    def cli_prompt(self):
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""

        print(f"{roles[1]}: ", end="")

        return inp


    def main(self,image_file):
        disable_torch_init()

        prompt = self.cli_prompt()
        self.start_chat(image_file,prompt)

        while True:
            prompt = self.cli_prompt()
            
            if not prompt:
                print("exiting...")
                break
            
            output = self.continue_chat(prompt)
            
            print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="BAAI/Bunny-v1_0-3B")
    #parser.add_argument("--model-base", type=str, default=None)
    #parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    #parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    #parser.add_argument("--load-8bit", action="store_true")
    #parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    bunny = BunnyHelper(
                    model_path=args.model_path,
                    device_map=args.device,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens)

    bunny.main(args.image_file)

