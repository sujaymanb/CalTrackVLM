import gradio as gr
import os
import pdb

from datetime import datetime
from entry import Entry, Meal, Macros
from model import Model

os.makedirs("static", exist_ok=True)
responses = []
model = Model()

def predict(image):
    if image is not None:
        filename = f"static/temp_entry.png"
        image.save(filename)
        parsed = model.run("", filename)
        print(parsed)
        new_entry = Entry(datetime.now())
        new_meal = Meal.from_dict(parsed)
        new_entry.meals.append(new_meal)
        return new_entry

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text"
)
app.launch(auth=("admin", "98Ho3i£3+i#|"), share=True)
