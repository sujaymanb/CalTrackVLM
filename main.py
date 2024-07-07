import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from bunny import Bunny

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

responses = []

# Model
model = Bunny()

def template(prompt):
    sub_command = 'Estimate calories and macro nutrients based on what the food is and the estimated amount of food in the picture.'
    text = f"A user asks an artificial intelligence diet assistant to help with their diet. The assistant gives accurate and concise estimates about the nutrietion content would based on photos of their meals. {sub_command} USER: <image>\n{prompt} \n\n ASSISTANT:"
    return text

def test_multi(image, prompt):
    text = template(prompt)
    return model.run(text, image)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "response": responses})

@app.post("/uploadimage/")
async def upload_image(image: UploadFile = File(...)):
    file_loc = f"static/{image.filename}"
    with open(file_loc, "wb+") as file:
        file.write(image.file.read())
    
    text = template("")
    output = model.run(text, image.filename)
    responses.append(output)
    
    return RedirectResponse(url="/", status_code=303)
