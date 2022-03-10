#Define the class
class Pyramid_Challenge():
    #activate the constructor
    def __init__(self, num):
        self.num = num #declare self.num

    def pyramid_challenge(self):
        """ Creates the pyramid from left to right. """
        self.number = self.num
        val1 = "#"
    
    #I made a mistake, before i used this formula (for number in range(number + 1))
        for number in range(self.num + 1): 
            print(val1*number)

pyramid_test = Pyramid_Challenge(6)
pyramid_test.pyramid_challenge()