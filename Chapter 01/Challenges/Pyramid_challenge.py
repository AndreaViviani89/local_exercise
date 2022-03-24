#Pyramid Challenge

#In this challenge you need to code a function that receives a number print out this pyramid:
#i. e. print_pyramid(5) outputs this:




def pyramid_challenge(num):

    val1 = "#"
    
    for number in range(num+1):
        print(val1*number)

pyramid_challenge(7)


#Different way with numbers ---> Specify the number of rows
rows = 5

#Probably the easiest way is the for loop
for i in range(rows):
    for a in range(i+1):
        print(a+1, end=" ")
    print("\n")

#I've discovered that I need to print \n for a new line


#Here I tried to create an input() function but without int() it returns an error

rows = int(input("Enter the number of rows: "))

for i in range(rows):
    for a in range(i+1):
        print(a+1, end=" ")
    print("\n")



