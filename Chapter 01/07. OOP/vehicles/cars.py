"""Group class exercise"""
"""I've created a class called cars with attributes and methods"""

class Vehicles:
    """1st - Create a constructor with different attributes"""
    def __init__(self, model, color, style, price, max_speed):
        self.model = model
        self.color = color
        self.style = style
        self.price = price
        self.max_speed = max_speed

    def change_price(self, price): #need to ask
        self.price = price 
    

    def calculate_vehicle_taxation(self, taxation):
        """Calculate the Italian vehicle taxation"""
        self.taxatin = taxation
        taxation_price = self.price * (1 + self.taxatin)
        return taxation_price


    def calculate_discount(self, discount):
        """Set discount formula"""
        self.discount = discount
        discount_price = self.price * (1 - self.discount)
        return discount_price

    # def calculate_luxury_vehicle(self, luxury):
    #     """I set a special tax for luxory vehicles"""
    #     self.luxury = luxury
    #     """I'd like to set an if - else statement but at the moment I've no clue"""
    #     luxory_taxation = self.price * (1 + self.luxury)
    #     return luxory_taxation

class Car (Vehicles):
    """I created a sub-class called Car inside the main class Vehicles"""
    def __init__(self, model, color, style, price, max_speed, luxury):
        super().__init__(model, color, style, price, max_speed)
        self.luxury = luxury

    """I set the double price formula for luxory cars. My goal was adapt an if - else statment"""
    def double_price(self):
        return self.price * 2

class Motorcycle (Vehicles):
    """I created a sub-class called Motorcycle inside the main class Vehicles"""
    def __init__(self, model, color, style, price, max_speed):
        super().__init__(model, color, style, price, max_speed)
    







    
