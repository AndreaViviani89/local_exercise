from collections import Counter
from itertools import product


def print_hi(name):
    print(f'This want be printed unless call, {name}')

if __name__ == '__main__': # __aaa__ private code
    print_hi("Hello World")


# ESEMPIO SU ABSTACTION
class Book(product):
    def __init__(self, name, publisher, isbn, selling_price=0): #"self" refering an instance of the class --> CONSTRUCTOR
        super().__init__(name, selling_price)
        self.publisher = publisher
        self.isbn = isbn

        if selling_price > 0:
            self.__selling_price = selling_price

    # ESEMPIO ENCAPSULATION
    @property
    # GETTER
    def selling_price(self):
        return self.__selling_prince

    @selling_price.setter
    def setting_price(self,price):
        if price <= 0:
            raise ValueError("The price must be greater than 0, we don't give away books in this shop")
        self.__selling_price = price 

    def __str__(self):
        return f'Book name:{self.name}\nBook Pubblisher: {self.publisher}\nBook ISBN: {self.isbn}\nBook Price: {self.selling_price}'