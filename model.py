from bunny import Bunny
from datetime import datetime
import ast

sub_command = """Given the picture of the meal above, identify individual food items in the image. 
For each item in the picture, give the following pieces of information in json format with a set of keys as follows:
{
    {
      "calories": 500,
      "macros": {
        "protein": 14.0,
        "carbs": 0.0,
        "sat_fat": 0.0,
        "unsat_fat": 0.0,
        "trans_fat": 0.0,
        "fiber": 0.0,
        "sugar": 0.0,
        "sodium": 0.0,
        "cholesterol": 0.0,
        "other": "n/a"
      }
    },
}

Output only the json and nothing else so that it is parseable with python's json parsing methods.
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
            #meal_data = ast.literal_eval(output)
            meal_data = json.loads(entry_json)

            # add date and user prompt
            meal_data['time_of_meal'] = datetime.now()
            meal_data['user_prompt'] = prompt

        except (TypeError, MemoryError, SyntaxError, ValueError): 
            error('could not parse output')

            # TODO reprompt to handle output issues

        return meal_data


'''test inference and parsing'''
model = Model()
image = './images/grillcheese.jpg'

parsed = model.run("", image)

print(parsed)