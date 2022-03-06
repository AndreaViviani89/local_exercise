

#Non Duplicated Challenge:
#In this challenge you need to code a function that receives a list of numbers and returns the non-duplicated number:

#i. e non_duplicated_challenge([1, 1, 2, 2, 3, 5, 5, 6, 6]) outputs 3
#i. e. non_duplicated_challenge([1, 2, 2, 3, 3]) outputs 1



def non_duplicated_challenge (list_of_numbers):   
    list_of_numbers = []

    for x in list_of_numbers:
        if list_of_numbers.count(x) > 1:
            pass
        else:
            list_of_numbers.append(x)
    return list_of_numbers

print(non_duplicated_challenge([1, 2, 2, 3, 3]))
print(non_duplicated_challenge([1, 1, 2, 2, 3, 5, 5, 6, 6]))

#Don't understand where is the error