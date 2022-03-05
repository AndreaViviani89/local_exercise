from queue import Empty


e_l = []
l = ["a", "b", "c", 1, 2, 3, True, False]
ones = [1, 1, 1]
'''
print( len(l))
print( len(e_l))
print (len(ones))
'''
for i in range(len(l)):
    print(l[i])
print(l[:4])

print(l[4:])

print(sum(ones))

e_l.append(5)
e_l.append(7)
print(e_l)
'''
e_l.remove(2)
print(e_l)
'''
'''
e_l.insert(2,1)
print(e_l)
'''
e_l.append(6)
e_l.append(9)

e_l.pop(2) #cancella un numero esatto dalla lista
print(e_l)