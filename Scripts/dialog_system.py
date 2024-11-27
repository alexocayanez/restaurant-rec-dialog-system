import re
import regex
import json

from Levenshtein import distance
import mlflow.pyfunc
import pyttsx3

from baselines import RuleBasedSystem
from recommendation import Recommendation


class DialogSystem:
    WELCOME_UTTERANCE = """Welcome to the Cambridge restaurant system! You can ask for restaurants by area , price range or food type . How may I help you?"""

    def __init__(self):
        PATH_RESTAURANT_INFO = """../Data/restaurant_info_new.csv"""
        self.recommendation = Recommendation(PATH_RESTAURANT_INFO)

        with open('../Config/config.json', 'r') as file:
            # Load the config file, where users can select preferences:
            self.config = json.load(file)
            # 1: All caps, 2: text_to_speech, 3:levenshtein_distance, 4: allow_restarts, 5: model

        self.previous_system_state = "init"
        self.FOODS = self.recommendation.FOOD_TYPES
        self.FOODS.update(["catalan", "steakhouse", "cuban", "spanish","swedish","world"])

        # Initialize engine tts model only if True 
        if self.config['text_to_speech']:
            self.engine = pyttsx3.init()
            # TODO: SMALL TWEAK: Allow user to configure volume (between 0 and 1)
            self.volume = self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('voice', self.engine.getProperty('voices')[0].id)

        # We set the model according config.json choice 
        model_name = self.config['model'].lower()
        if model_name == 'logisticregression' or model_name == 'logistic_regression':
            self.model = mlflow.pyfunc.load_model(model_uri="../Models/log_res_model").unwrap_python_model()
        elif model_name == 'decisiontree' or model_name == 'decision_tree':
            self.model = mlflow.pyfunc.load_model(model_uri="../Models/tree_model").unwrap_python_model()
        elif model_name == "baseline":
            self.model = RuleBasedSystem()
        else:
            self.model = None

    def state_transition(self, recommendation: Recommendation, previous_system_state: str, user_utterance: str,
                         user_state: str, debug=False):
        """
        State transition function of the dialog system. Outputs next system state and a corresponding system utterance
        """

        # If the dialog is classified as goodbye the conversation ends.
        if user_state == "bye":
            return 'ending', ''

        if recommendation.add_requirements and user_state != "inform":
            recommendation.add_requirements = False
        # If the dialog id classified as restart we restart the parameters we are considering in the transition and
        # the recommendation search.
        if user_state == "restart" and self.config['allow_restarts']:
            recommendation.reset()
            return 'init', 'Restarting...'
        ###   STATE TRANSITION DIAGRAM STARTS  ###

        # Retrieving user preferences
        keywords = self.keyword_matching(user_utterance)
        pattern_values = self.pattern_matching(user_utterance)
        if debug:
            print("##### (INFORM TAG) Matched keywords: " +
                  str(keywords) + " #####")
            print("##### Matched keywords in patterns: " +
                  str(pattern_values) + " #####")

        # If dialog act is classified as inform it is ok for the system to retrieve preferences for exact matches.
        if user_state == "inform":
            if keywords['food']:
                recommendation.food_type = keywords['food']
            if keywords['area']:
                recommendation.area = keywords['area']
            if keywords['price']:
                recommendation.price = keywords['price']
            if keywords['requirement']:
                recommendation.add_requirements = True
                recommendation.req = keywords['requirement']
            if keywords['reasoning'] is not None:
                recommendation.tf = keywords['reasoning']

        # We can look for keywords through common patterns

        if pattern_values['food']:
            recommendation.food_type = pattern_values['food']
        if pattern_values['area']:
            recommendation.area = pattern_values['area']
        if pattern_values['price']:
            recommendation.price = pattern_values['price']

        # If we detect that the user is expressing no preference, we check if the system asked a question.
        if keywords['any']:
            if previous_system_state == "askarea":
                recommendation.area = 'any'
            if previous_system_state == "askfoodtype":
                recommendation.food_type = 'any'
            if previous_system_state == "askpricerange":
                recommendation.price = 'any'

        # PART 1 : RECOMMENDATION'S LOOP, we need to keep retrieving preferences until we find a recommendation in
        # the dataset
        if not recommendation.retrieved:

            # STATE 2) If area is not stablished it will be asked
            if not recommendation.area and recommendation.area != 0:
                system_state = "askarea"
                system_utterance = """What part of town do you have in mind?"""
                return system_state, recommendation.current_search_to_str() + ' ' + system_utterance

            # STATE 3) If food type is not stablished it will be asked
            if not recommendation.food_type and recommendation.food_type != 0:
                system_state = "askfoodtype"
                system_utterance = """What kind of food would you like?"""
                return system_state, recommendation.current_search_to_str() + ' ' + system_utterance

            # STATE 4) If price range is not stablished and there are still more than 1 option available, it will be
            # asked.
            if recommendation.is_not_concluded() and not recommendation.price and recommendation.price != 0:
                system_state = "askpricerange"
                system_utterance = """Would you like something in the cheap , moderate , or expensive price range?"""
                return system_state, recommendation.current_search_to_str() + ' ' + system_utterance

            # STATE 5) Asking additional requirements
            if not recommendation.additional_requirements_asked:
                recommendation.additional_requirements_asked = True
                system_state = "askrequirements"
                system_utterance = "Do you have any additional requirements?"
                return system_state, system_utterance

            if debug:
                print(
                    f"##### Performing search with: area={recommendation.area}, price={recommendation.price}, food={recommendation.food_type}, requirement={recommendation.req} #####")

            # STATE 6) Retrieving preferences again if there is no match
            if len(recommendation.search().index) == 0:
                system_state = "nomatch"
                recommendation.reset()
                system_utterance = f"There is no restaurant in the city with the specified preferences. Restarting..."
                return system_state, system_utterance

            # STATE 7) Reply option for restaurant
            else:
                if user_state == "negate":  # If there were no additional requirements:
                    recommendation.add_requirements = False
                recommendation.retrieved = True
                recommendation.recommendation_index = 0
                system_state = "recommending"
                return system_state, recommendation.current_k_reccomendation_to_str(recommendation.recommendation_index)

        # PART 2: GIVING INFORMATION ABOUT THE RECOMMENDATION
        if recommendation.retrieved:

            # STATE 7b) Give alternative if the user asks for it.
            if user_state in {"reqmore", "reqalts"}:
                system_state = "recommending"
                if len(recommendation.search()) == 1:
                    system_utterance = 'There are no more alternatives with set preferences.'
                else:
                    system_utterance = ''
                    recommendation.recommendation_index += 1
                    if recommendation.recommendation_index == len(recommendation.search()):
                        recommendation.recommendation_index = 0
                        system_utterance += "All options shown, returning to the first one. "
                    system_utterance += recommendation.current_k_reccomendation_to_str(
                        recommendation.recommendation_index)
                return system_state, system_utterance

            # STATE 8) Give requested information about recommendation if user asks for it.
            if user_state == "request":
                system_state = "providinginformation"
                reco = recommendation.current_k_reccomendation(
                    recommendation.recommendation_index)
                requested_parameters = []
                for word in user_utterance.split():
                    if word in {'phone', 'telephone'}:
                        requested_parameters.append('phone')
                    if word in {'addr', 'addre', 'address', 'where'}:
                        requested_parameters.append('address')

                    # TODO: Fix postcode and food_type
                    if word in {'post', 'postal', 'mail', 'postcode'}:
                        requested_parameters.append('postcode')
                    if word in {'type', 'kind', 'serve', 'food'}:
                        requested_parameters.append('food_type')

                system_utterance = ''
                parameter_names = {'phone': 'telephone number', 'address': 'address',
                                   'postcode': 'post code', 'food_type': 'type of food'}
                for i, parameter in enumerate(requested_parameters):
                    if i == 0:
                        system_utterance += f"""The {parameter_names[parameter]} of the {reco['name']} is {reco[parameter]}"""
                    else:
                        system_utterance += f" and its {parameter_names[parameter]} is {reco[parameter]}"
                system_utterance += '.'
                return system_state, system_utterance

        # EXCEPTION STATE If nothing happens
        if previous_system_state == "NOT IN THE TRANSITION" and user_state == "negate":
            system_state = 'ending'
            return system_state, "Exiting the application. Goodbye!"

        if previous_system_state == "NOT IN THE TRANSITION":
            system_state = 'init'
            recommendation.reset()
            return system_state, "Restarting the application..."

        system_state = 'NOT IN THE TRANSITION'
        system_utterance = """Do you want to search again? If no, program will exit"""
        return system_state, system_utterance

    def give_utterance(self, system_utterance: str):
        if self.config['all_caps']:
            print("SYSTEM: " + system_utterance.upper())
        else:
            print("System: " + system_utterance)

        if self.config['text_to_speech'] == True:
            self.engine.say(system_utterance)
            self.engine.runAndWait()

    def find_closest_levensthein(self, word: str, possible_values: list[str], max_distance: int):
        """
        Find closest word using Levenshtein distance
        """

        match = None
        for value in possible_values:
            Levensthein_distance = distance(word, value)
            if Levensthein_distance <= max_distance:
                max_distance = Levensthein_distance
                match = value
        return match

    def keyword_matching(self, user_utterance):
        """
        Function that returns a dictionary {'area': ..., 'price': ..., 'food': ..., 'any': True/False}.
        'any' is true when 'any' (or something with the same meaning) is found in the user utterance.
        It is then up to the DialogManager to determine what this 'any' means.
        Regex package needed instead of re package to enable Levenshtein distance search.
        """
        user_utterance = user_utterance.lower()
        data = {'area': None, 'price': None, 'food': None,
                'any': False, 'requirement': None, 'reasoning': None}

        # Check for any
        any_pattern = r"\b(any|anywhere|anything|don't care)\b"
        match = re.search(any_pattern, user_utterance)
        if match:
            data['any'] = True

        # Check for area
        areas = ['centre', 'center', 'west', 'east', 'south', 'north']
        for area in areas:
            area_pattern = fr"(\b{area}\b)" + r"{e<=1,s<=1}"
            match = regex.search(area_pattern, user_utterance)
            if match:
                data['area'] = area if area != 'center' else 'centre'
                break

        # Check for price
        prices = ['expensive', 'moderate', 'cheap']
        for price in prices:
            area_pattern = fr"(\b{price}\b)" + r"{e<=3,s<=2}"
            match = regex.search(area_pattern, user_utterance)
            if match:
                data['price'] = price
                break

        # Check for food type. No levenshtein substitutions here because some words are just too similar
        # For example 'thai' - 'that'
        for food in self.FOODS:
            area_pattern = fr"(\b{food}\b)" + r"{e<=3,s<=0}"
            match = regex.search(area_pattern, user_utterance)
            if match:
                data['food'] = food
                break

        # Check for additional requirements
        requirements = ['touristic', 'assigned seats', 'children', 'romantic']
        for req in requirements:
            req_pattern = fr"(\b{req}\b)" + r"{e<=3,s<=2}"
            match = regex.search(req_pattern, user_utterance)
            if match:
                data['requirement'] = req
                data['reasoning'] = True
                # Check for negation in additional requirements
                for negation in ['no', 'not']:
                    if negation in user_utterance.split():
                        data['reasoning'] = False
                break
        return data

    def pattern_matching(self, user_utterance):
        """
        Function used for retrieving user's preferences. Stores the preferences in self.request_info.
        When debug is set to True, the function prints the utterance and results to the terminal.
        """
        data = {'area': None, 'price': None, 'food': None, 'any': False}

        # AREA
        # Does not yet account for typo's in location not sure if that is required.
        AREA_PATTERN = r"centre|west|south|north|east|(any) area|(any) part of town"
        match = re.search(AREA_PATTERN, user_utterance)
        if match:
            area = match.group(1) or match.group(0)
            data["area"] = area

        # FOOD_TYPE
        FOOD_TYPE_PATTERNS = [
            # Tests for serves and takes the food variable after it, for example "Serves Catalan food"
            r"serves\s+(\w+)",
            # Same but for serving
            r"serving\s+(\w+)",
            # Same but the word inbetween looking for and food "I am looking for Catalan food"
            r"looking\s+for\s+(\w+)\s+food",
            # If the food type appears before food e.g "Catalan food"
            r"(\w+)\s+food",
            # If the food type is surrounded by "Need a", "Restaurant" such as "I need a Cuban restaurant"
            r"need\s+a\s+(\w+)\s+restaurant",
            # Same but find a ... restaurant instead. such as "Find a Cuban restaurant"
            r"find\s+a\s+(\w+)\s+restaurant"
        ]
        food_type_matched = None
        for pattern in FOOD_TYPE_PATTERNS:
            match = re.search(pattern, user_utterance)

            if match:
                # First group is the food_type
                food_type = match.group(1).strip()
                if food_type != "cheap":  # Regex function accepted "cheap" as a food_type.
                    # Find matches within 3 Levenshtein distance
                    food_type_matched = self.find_closest_levensthein(
                        food_type, self.FOODS, max_distance=self.config['levenshtein_distance'])
                    print(food_type_matched)
                    data["food"] = food_type_matched
                    break

        # PRICE
        PRICE_PATTERN = r"cheap|moderate|expensive|(any) price range"
        match = re.search(PRICE_PATTERN, user_utterance)
        if match:
            price = match.group(1) or match.group(0)
            data["price"] = price

        return data

    def run(self, debug=False):
        previous_system_state = "init"
        k = 0
        while True:
            if previous_system_state == 'init':
                self.give_utterance(self.WELCOME_UTTERANCE)
            if previous_system_state != "nomatch":
                user_utterance = input().lower()
            if user_utterance == 'exit':
                break
            elif user_utterance == 'restart' and self.config['allow_restarts']:
                user_state = 'restart'
            else:
                user_state = self.model.predict_tag(user_utterance)
            if debug:
                print('--------------------------------------')
                print('User state: ' + user_state)

            system_state, system_utterance = self.state_transition(self.recommendation,
                                                                   previous_system_state,
                                                                   user_utterance,
                                                                   user_state,
                                                                   debug)
            if system_state == 'ending':
                self.give_utterance("Exiting the application. Goodbye!")
                break

            self.give_utterance(system_utterance)
            if debug:
                print('System state: ' + system_state)
                print('--------------------------------------')

            previous_system_state = system_state
            k += 1
        return


def main():
    debug = False
    dialog_system = DialogSystem()
    dialog_system.run(debug)


if __name__ == "__main__":
    main()
