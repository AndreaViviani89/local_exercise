
#is not going to be define


from abc import ABC, abstractmethod

class Polygon(ABC):
    @abstractmethod
    def get_number_of_sides(self):
        pass

    def print_me(self):
        print('Im a polygon')


class Triangle(Polygon):
    def get_number_of_sides(self):
        return 3

        
