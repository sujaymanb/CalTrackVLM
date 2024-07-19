'''
The data model for a daily entry into the journal.

Example:

Entry:
├── Date of meal: 07-19-2024
└── Meals:
        ├── Time: 04-01 PM
        ├── Original prompt: grilled chicken
        ├── Calories: 500
        └── Macros: {'protein': 14.0, 'carbs': 0.0, 'sat_fat': 0.0, 'unsat_fat': 0.0, 'trans_fat': 0.0, 'fiber': 0.0, 'sugar': 0.0, 'sodium': 0.0, 'cholesterol': 0.0, 'other': 'n/a'}
        │
        ├── Time: 04-01 PM
        ├── Original prompt: rice and beans
        ├── Calories: 700
        └── Macros: {'protein': 0.0, 'carbs': 0.0, 'sat_fat': 0.0, 'unsat_fat': 0.0, 'trans_fat': 0.0, 'fiber': 0.0, 'sugar': 0.0, 'sodium': 0.0, 'cholesterol': 0.0, 'other': 'n/a'}
'''


import json
from datetime import datetime
from typing import List, Dict, Optional, TypedDict

class Macros(TypedDict):
    protein: float
    carbs: float
    sat_fat: float
    unsat_fat: float
    trans_fat: float
    fiber: float
    sugar: float
    sodium: float
    cholesterol: float
    other: str

def default_macros() -> Macros:
    return {"protein": 0.0, "carbs": 0.0, "sat_fat": 0.0, "unsat_fat": 0.0, "trans_fat": 0.0, "fiber": 0.0, "sugar": 0.0, "sodium": 0.0, "cholesterol": 0.0, "other": "n/a"}

class Meal:
    def __init__(self, time_of_meal: datetime, user_prompt: str, calories: Optional[float] = 0, macros: Optional[Macros] = None) -> None:
        self.time_of_meal: datetime = time_of_meal
        self.user_prompt: str = user_prompt
        self.calories: float = calories

        # init macros
        default_macro_values = default_macros()
        if macros is None:
            # default macros
            self.macros = default_macro_values
        else:
            # overwrite default macros with partial/all provided macros
            self.macros = Macros(**{**default_macro_values, **macros})
    
    def to_dict(self) -> Dict:
        return {
            "time_of_meal": self.time_of_meal.isoformat(),
            "user_prompt": self.user_prompt,
            "calories": self.calories,
            "macros": self.macros
        }
    
    @classmethod
    def from_dict(cls, json_data: Dict) -> 'Meal':
        return cls(
            time_of_meal=datetime.fromisoformat(json_data["time_of_meal"]),
            user_prompt=json_data["user_prompt"],
            calories=json_data.get("calories", 0),
            macros=json_data.get("macros", default_macros())
        )
    
    def __repr__(self) -> str:
        return (f"\t├── Time: {self.time_of_meal.strftime("%I-%M %p")}\n\t├── Original prompt: {self.user_prompt}\n\t├── Calories: {self.calories}\n\t└── Macros: {self.macros}\n")

class Entry:
    def __init__(self, date: datetime, meals: Optional[List[Meal]] = list()) -> None:
        self.date: datetime = date
        self.meals: List[Meal] = meals

    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "meals": [meal.to_dict() for meal in self.meals]
        }
    
    @classmethod
    def from_dict(cls, json_data: Dict) -> 'Entry':
        return cls(
            date=datetime.fromisoformat(json_data["date"]),
            meals=[Meal.from_dict(meal) for meal in json_data.get("meals", [])] 
        )

    def __repr__(self) -> str:
        meals = "\t│\n".join(repr(meal) for meal in self.meals)
        return (f"Entry:\n├── Date of meal: {self.date.strftime("%m-%d-%Y")}\n└── Meals:\n{meals}\n")

# Example
# Serializing and deserializing data to and from JSON
# Create a new entry consisting of two meals
entry = Entry(date=datetime.now(), meals=[
    Meal(time_of_meal=datetime.now(), user_prompt="grilled chicken", calories=500, macros=Macros(protein=14.0)),
    Meal(time_of_meal=datetime.now(), user_prompt="rice and beans", calories=700)
])

# Serialize Entry object into JSON string
entry_json = json.dumps(entry.to_dict(), indent=4)
print(entry_json)

# Deserialize JSON string into Entry object
entry_dict = json.loads(entry_json)
new_entry = Entry.from_dict(entry_dict)
print(new_entry)