from bunny import Bunny
from datetime import datetime
import ast
import json
from entry import Entry, Meal, Macros

sub_command = """Given the picture of the meal above, identify individual food items in the image. 
For each item in the picture, give values to each key in json format using this template:
{
    "calories":,
    "macros": {
    "protein":,
    "carbs":,
    "sat_fat":,
    "unsat_fat":,
    "trans_fat":,
    "fiber":,
    "sugar":,
    "sodium":,
    "cholesterol":,
    }
},

Output only the json and nothing else so that it is parseable with python's json parsing methods.
    """

PRE_PROMPT = """Given the picture of the meal above, identify and name each food item. If different food items are components of one whole dish, do not separate them as different food items. 

Output them into a list.
    """

class Model:
    '''Model class that abstracts which model we are running'''
    def __init__(self, model_type = 'bunny'):

        # to allow multiple model types
        if model_type == 'bunny':
            self.model = Bunny()

    def pre_prompt_template(self, pre_prompt):
        text = f"You are a diet tracking assistant. USER: <image>\n{pre_prompt}\nASSISTANT:"
        return text
    
    def template(self, prompt):
        text = f"You are a diet tracking assistant. USER: <image>\n{prompt}\n{sub_command}\nASSISTANT:"
        return text

    def run(self, prompt, image):
        meal_data = None

        pre_prompt_output = self.model.run(self.pre_prompt_template(PRE_PROMPT), image)
        print(pre_prompt_output)

        text = self.template(prompt)
        output = self.model.run(text, image)
        
        # try parse
        try:  
            #meal_data = ast.literal_eval(output)
            meal_data = json.loads(output)

            # add date and user prompt
            meal_data['time_of_meal'] = datetime.now()
            meal_data['user_prompt'] = prompt
            meal_data['meal_name'] = pre_prompt_output

        except (TypeError, MemoryError, SyntaxError, ValueError) as e: 
            print(e)

            # TODO reprompt to handle output issues

        return meal_data

if __name__ == "__main__":
    '''test inference and parsing'''
    model = Model()
    image = './images/grillcheese.jpg'
    parsed = model.run("", image)
    print(parsed)

    new_entry = Entry(datetime.now())
    new_meal = Meal.from_dict(parsed)
    new_entry.meals.append(new_meal)
    print(new_entry)