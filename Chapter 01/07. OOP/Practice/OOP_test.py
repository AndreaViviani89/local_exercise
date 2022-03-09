from collections import Counter


def print_hi(name):
    print(f'This want be printed unless call, {name}')

if __name__ == '__main__': # __aaa__ private code
    print_hi("Hello World")


# ESEMPIO SU ABSTACTION
class Book:
    def __init__(self, name, publisher, isbn, selling_price=0): #"self" refering an instance of the class --> CONSTRUCTOR
        self.name = name
        self.publisher = publisher
        self.isbn = isbn
        self.__ratings_stars = []

        self.__selling_price = selling_price

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


    def add_rating(self, stars):
        self.__ratings_stars.apppend(stars)

    def get_ratings_average(self):
        return sum(self.__ratings.stars) / len(self.__ratings_stars)

    def get_total_ratings(self):
        return len(self.__ratings_stars)

    def get_stars_count(self):
        return dict(Counter(self.__ratings_stars))

