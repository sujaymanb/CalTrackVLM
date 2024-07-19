import dataclasses
import base64
from io import BytesIO

@dataclasses.dataclass
class Conversation:
	"""keeps convo history"""
	system_message: str
	roles: list[str]
	messages: list[list[str]]
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
text = f"You are a diet tracking assistant. USER: <image>\n{prompt} ASSISTANT:"

conv_template = Conversation(
		system_message=text,
		roles=("USER", "ASSISTANT"),
		messages=(),
		seperator=" ",
		end="<|endoftext|>"
	)

if __name__ == "__main__":
	print(conv_template.get_prompt())
