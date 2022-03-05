var1 = "Hello \n world"
var2 = 'Hello \t world'
var3 = """
        Hello World
        """
var4 = 'Hello "to" the world'
print(var1)
print(var2)
print(var3)
print(var4)
print(var2[:5], var4[6:])

for i in range( len(var4) ):
    print(var4[i])

for i in range( len(var4) ):
    print(var4[i]+ "!")

for i in range( len(var4) ):
    print(var4[i] * 4)