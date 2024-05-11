import gradio as gr
from bunny import Bunny

# load VLM
model = Bunny()

def template(prompt):
    """baseline template"""
    sub_command = 'Estimate calories and macro nutrients based on what the food is and the estimated amount of food in the picture.'
    text = f"A user asks an artificial intelligence diet assistant to help with their diet. The assistant gives accurate and concise estimates about the nutrition content would based on photos of their meals. {sub_command} USER: <image>\n{prompt} ASSISTANT:"

    return text

def test_multi(image, prompt):
    """demo function calls model run function"""
    text = template(prompt)
    return model.run(text,image)

# Simple demo
image = gr.Image(shape=(224, 224))
demo = gr.Interface(
            fn=test_multi,
            inputs=[image,"text"],
            outputs="text"
        )

demo.launch()