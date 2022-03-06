#Inverted pyramed challenge

#In this challenge you need to code a function that receives a number print out this pyramid:
#i. e. print_pyramid(5) outputs this:



def inverted_pyramed_challenge(num):
    i = num
    while (i <= num):
        print("#" * i)
        i -= 1
        if i == 0:
            break
    return inverted_pyramed_challenge   
    

print(inverted_pyramed_challenge(6)) 

