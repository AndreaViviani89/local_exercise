# Strive School Challenge
#In this challenge you need to code a function that receives a list of numbers and returns:

'''
    Strive if the number is divisible by 3
    School if the number is divisible by 5
    Strive School if the number is divisible by 3 and 5
    the number itself otherwise

i. e. strive_school([1, 2, 3, 5, 15]) outputs [1, 2, Strive, School, Strive School]

'''




def strive_school(list_of_numbers):
    list_empty = []
    for x in list_of_numbers:
        if x % 3 == 0:
            list_empty.append("Strive")
        elif x % 5 == 0:
            list_empty.append("School")
        elif x % 3 == 0 and x % 5 == 0:
            list_empty.append("Strive School")
        else:
            list_empty.append(x)
    return list_empty


list_of_numers = [1, 3, 5, 15, 7, 34, 55, 66, 90]
list_of_numers.append(300)
print(strive_school(list_of_numers))




