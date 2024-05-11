import gradio as gr
from transformers import pipeline

model = pipeline("text-generation")


def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

def test_multi(image, prompt):
    return "prompt: " + prompt

#demo = gr.Interface(fn=predict, inputs="text", outputs="text")

image = gr.Image(shape=(224, 224))

demo = gr.Interface(
            fn=test_multi,
            inputs=[image,"text"],
            outputs="text"
        )


demo.launch()