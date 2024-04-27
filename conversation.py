import dataclasses
import base64
from io import BytesIO

@dataclasses.dataclass
class Conversation:
	"""keeps convo history"""
	system_message: str,
	roles: List[str]
	messages: List[List[str]]
	seperator: str = " "
	end: str = "<|endoftext|>"

	def get_prompt(self):
		if len(self.messages) < 1:
			return []

		messages = self.messages.copy()
		init_role, init_msg = messages[0].copy()
		init_msg = init_msg[0].replace("<image>","").strip()
		
		messages[0] = (init_role, "<image>\n" + init_msg)

		seps = [self.seperator, self.end]
		history = self.system_message + seps[0]

		for i, (role, message) in enumerate(messages):
			if message:
				if type(message) is tuple:
					message, _ = messages
				history += role + ": " + message + seps[i % 2]
			else:
				history += role + ":"

		return history

	def append_message(self, role, message):
		self.messages.append([role, message])

	def get_images(self, return_pil=False):
		images = []
		for i, (role, msg) in enumerate(self.messages):
			if i % 2 == 0:
				if type(msg) is tuple:
					msg, image = msg

				max_hw, min_hw = max(image.size), min(image.size)
				aspect_ratio = max_hw/min_hw
				max_len, min_len = 800, 400
				shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
				longest_edge = int(shortest_edge * aspect_ratio)

				W, H = image.size

				if longest_edge != max(image.size):
					if H > W:
						H, W = longest_edge, shortest_edge
					else:
						H, W = shortest_edge, longest_edge
					image = image.resize((W, H))

				buffered = BytesIO()
					image.save(buffered, format="PNG")
					img_b64_str = base64.b64encode(buffered.getvalue()).decode()
					images.append(img_b64_str)

		return images



conv_template = Conversation(
		system_message="A user asks an artificial intelligence diet assistant to help with their diet. The assistant gives accurate and concise estimates about the nutrition content would based on photos of their meals. USER: <image>\n ASSISTANT:",
		roles=("USER", "ASSISTANT"),
		messages=(),
		seperator=" ",
		end="<|endoftext|>"
	)

if __name__ == "__main__":
	print(conv_template.get_prompt())
