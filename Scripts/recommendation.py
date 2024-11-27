from typing import Optional, Any
import pandas as pd
import numpy as np

class Recommendation:
    def __init__(self, restaurant_info_path="../Data/restaurant_info_new.csv") -> None:
        self.path = restaurant_info_path
        self.data = pd.read_csv(self.path)
        self.PRICE_RANGES = set(self.data['pricerange'].unique())
        self.AREAS = set(self.data['area'].unique())
        self.FOOD_TYPES = set(self.data['food'].unique())

        self._food_type = None
        self._price = None
        self._area = None
        self._req = None
        self._tf = None

        self.retrieved = False
        self.additional_requirements_asked = False
        self.add_requirements = False
        self.recommendation_index = -1

    ### PROPERTIES SETTING ### 
    # This is done to allow to set in the state transition function any property to 'any' or 'dont care', 
    # so it is still set to 0 allowing the proper functioning of the transition modelling.

    @property
    def food_type(self):
        return self._food_type

    @food_type.setter
    def food_type(self, value):
        if value == "any" or value == "dontcare":
            self._food_type = 0
        else:
            self._food_type = value

    @food_type.deleter
    def food_type(self):
        del self._food_type

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        if value == "any" or value == "dontcare":
            self._price = 0
        else:
            self._price = value

    @price.deleter
    def price(self):
        del self._price

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value):
        if value == "any" or value == "dontcare":
            self._area = 0
        else:
            self._area = value

    @area.deleter
    def area(self):
        del self._area

    @property
    def req(self):
        return self._req

    @req.setter
    def req(self, value):
        self._req = value

    @req.deleter
    def req(self):
        del self._req

    @property
    def tf(self):
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = value

    @tf.deleter
    def tf(self):
        del self._tf

    ### LOOK UP FUNCTIONS ###

    def is_not_concluded(self) -> bool:
        """
        Function that provides a True value if the current search has still 2 or more restaurants and a False value otherwise.
        """
        if len(self.search()) <= 1:
            return False
        return True

    def search(self) -> pd.DataFrame:
        """
        Lookup function on the database using the current stated food_type, price and area properties
        """
        df = self.data
        mask = [True] * len(df)
        if self.food_type:
            mask &= (df['food'] == self.food_type)
        if self.area:
            mask &= (df['area'] == self.area)
        if self.price:
            mask &= (df['pricerange'] == self.price)
        df = df[mask]
        if self.add_requirements:
            df = self.add_additional_properties(df)
            df = df.dropna(subset=[self.req])
            df = df[df[self.req]==self.tf]
        return df

    def current_search_to_str(self) -> str:
        """
        Function that returns a string specifying the number of restaurants found with the current properties. 
        """
        search = self.search()
        if len(search) == 0:
            return "There are no restaurants in the city with the required parameters."
        if self._food_type and self._price and self._area:
            return f"""There are {len(search)} restaurants serving {self._food_type} food in the {self._area} of the town and the {self._price} price range. """
        elif self._food_type and self._price:
            return f"""There are {len(search)} restaurants serving {self._food_type} food in the {self._price} price range. """
        elif self._food_type and self._area:
            return f"""There are {len(search)} restaurants serving {self._food_type} food in the {self._area} of the town. """
        elif self._area and self._price:
            return f"""There are {len(search)} restaurants in the {self._area} of the town and the {self._price} price range. """
        elif self._food_type:
            return f"""There are {len(search)} restaurants serving {self._food_type} food. """
        elif self._price:
            return f"""There are {len(search)} restaurants in the {self._price} price range. """
        elif self._area:
            return f"""There are {len(search)} restaurants in the {self._area} of the town. """
        else:
            return ''

    def current_k_reccomendation(self, k: int) -> Optional[dict[str, Any]]:
        """
        Function that takes as input the recommendation index on the current search and gives a dictionary with its information.
        """
        search = self.search()
        if self.add_requirements:
            search = self.add_additional_properties(search)
            search = search.dropna(subset=[self.req])
            search = search[search[self.req]==self.tf]
        if k < len(search):
            reco = search.iloc[k]
            return {'name': reco['restaurantname'],
                    'price': reco['pricerange'],
                    'area': reco['area'],
                    'food_type': reco['food'],
                    'phone': reco['phone'],
                    'address': reco['addr'],
                    'postcode': reco['postcode']
                    }
        else:
            return None

    def current_k_reccomendation_to_str(self, k: int) -> str:
        """
        Function that takes as input the recommendation index on the current search and gives a string with its information.
        """
        reco = self.current_k_reccomendation(k)

        if self.req == "touristic" and self.tf:
            reason = f"\n The restaurant is touristic because it is a cheap restaurant with good food quality"
        elif self.req == "touristic" and not self.tf:
            reason = f"\n The restaurant is not touristic because Romanian cuisine is unknown for most tourists and they prefer familiar food"
        elif self.req == "assigned seats":
            reason = f"\n The restaurant has assigned seats because the restaurant is busy and the waiter decides where you sit"
        elif self.req == "children" and not self.tf:
            reason = f"\n The restaurant is not good for children spending a long time is not advised when taking children"
        elif self.req == "romantic" and self.tf:
            reason = f"\n The restaurant is romantic because it allows you to stay for a long time"
        elif self.req == "romantic" and not self.tf:
            reason = f"\n The restaurant is not romantic because a busy restaurant is not romantic"
        else:
            reason = ""
        return f"""{reco['name']} is a great restaurant serving {reco['food_type']} in the {reco['area']} part of the city at {reco['price']} price.{reason}"""

    def add_additional_properties(self, search_df) -> pd.DataFrame:
        # Set defaults
        df = search_df.copy()
        df['touristic'] = np.nan
        df['assigned seats'] = np.nan
        df['children'] = np.nan
        df['romantic'] = np.nan
        # Apply rules
        rule1_condition = (df['pricerange'] == 'cheap') & (df['food quality'] == 'good')
        df.loc[rule1_condition, 'touristic'] = True
        rule2_condition = (df['food'] == 'romanian')
        df.loc[rule2_condition, 'touristic'] = False
        rule3_condition = (df['crowdedness'] == 'busy')
        df.loc[rule3_condition, 'assigned seats'] = True
        rule4_condition = (df['length of stay'] == 'long')
        df.loc[rule4_condition, 'children'] = False
        rule5_condition = (df['crowdedness'] == 'busy')
        df.loc[rule5_condition, 'romantic'] = False
        rule6_condition = (df['length of stay'] == 'long')
        df.loc[rule6_condition, 'romantic'] = True
        return df

    # Reset function
    def reset(self):
        setattr(self, 'retrieved', False)
        setattr(self, 'recommendation_index', -1)
        self.food_type = None
        self.area = None
        self.price = None
        self.req = None
        self.additional_requirements_asked = False
        self.add_requirements = False
        self.retrieved = False
