'''

from unicodedata import name
from Practice.product import Product


class Magazine(Product):
    def __init__(self, name, selling_prince = 0):
        super().__init__(name, selling_price)
        if selling_price >= 0:
            self.__selling_price = selling_price
        
    
    @property
    def selling_price(self):
        if self.__selling_price == 0:
            return 'This magazine is free, enjoy it!'
        return self.__selling_price
    
    @selling_price.setter
    def setting_price(self,price):
        if price <= 0:
            raise ValueError("The price cannot be negative, we don't pay customers to take our free magazines")
        self.__selling_price = price 
'''