from bunny import Bunny
import ast

sub_command = """Given the picture of the meal above, identify individual food items in the image. 
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

class Model:
    '''Model class that abstracts which model we are running'''
    def __init__(self, model_type = 'bunny'):

        # to allow multiple model types
        if model_type == 'bunny':
            self.model = Bunny()

    def template(self, prompt):
        text = f"You are a diet tracking assistant. USER: <image>\n{prompt}\n{sub_command}\nASSISTANT:"
        return text

    def run(self, prompt, image):
        meal_data = None

        text = self.template(prompt)
        output = self.model.run(text, image)
        
        # try parse
        try:  
            meal_data = ast.literal_eval(output)
        except (TypeError, MemoryError, SyntaxError, ValueError): 
            error('could not parse output')

            # TODO reprompt to handle output issues

        return meal_data


'''test inference and parsing'''
model = Model()
image = './images/grillcheese.jpg'

parsed = model("", image.filename)

print(parsed)