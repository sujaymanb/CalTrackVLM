


class CalTrackBot:
    def __init__(self,
            model_path: str = 'BUNNY TODO',
            device_map: str = 'auto',
            ) -> None:
        self.model = None
        self.tokenizer = None
        self.img_tensor = None
        self.conv_history = None
        self.conv_img = None

        self.load_models(model_path,
                        device_map=device_map)

    def load_models(self, model_path: str,
                    device_map: str) -> None:
        """ Load the model and tokenizer """
        self.model = BunnyForCausalLM.from_pretrained(model_path,
                                                    device_map=device_map)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                        trust_remote=True)

        # TODO other loading stuff

    def process_image(self, img_path: str) -> None:
        """ Load and process the image """
        # TODO call model.process iamge
        self.conv_img = PIL.load(img_path)
        self.image_tensor = None # TODO

    def generate_answer(self, **kwargs) -> str:
        """ Generate an answer from the current conversation """
        raw_prompt = self.conv.get_prompt()
        input_ids = tokenizer() # TODO

        # TODO generate output ids

        return output.split

    def get_conv_text(self) -> str:
        """ return full conversation text """
        return self.conv.get_prompt()

    def start_chat(self,
                img_path: str,
                prompt: str,
                max_new_tokens=1024,
                **kwargs) -> str:
        """ start a new convo/session/meal """
        self.process_image(img_path)
        self.conv = conv_template.copy()
        
        answer = self.generate_answer()

        return answer

    def continue_chat(self,
                    prompt: str,
                    max_new_tokens=1024,
                    **kwargs) -> str:
        """ chain/continue existing conversation """
        if self.conv is None:
            raise RuntimeError("No existing conversation text.")

        self.conv.append_message("USER", prompt)
        self.conv.append_message("ASSISTANT", None)
        answer = self.generate_answer()

        return answer



