#per aprire un file
'''
f = open("example.txt", "w") #"w" --> write (creare un file) ___ "r" --> read (aprire un file) PS. se questo non esiste si verificherà un errore
'''


'''
#miglior modo per aprire/creare/leggere un file
with open("example.txt", "w") as f:
    #here the file is open
    f.write("Hello World")

#here the file is close
f.write("!")
'''

'''
#altro modo utile per i file
with open("example.txt", "w") as f:
    f.writelines(["Hello\n", " World", "!"])
'''
'''
with open("example.txt", "a") as f: #"a" --> append
    f.write("!!!!!!!")
'''

'''
with open("example.txt", "r") as f: # "r" --> read --> only read NOT write
    text = f.read()
    print(text)
'''
'''
with open("example.txt", "r") as f:
    text1 = f.readline()
    text2 = f.readline()
    print(text1)
    print(text2)
'''
with open("example.txt", "r") as f:
    text = f.readlines()
    print(text)

with open("example.txt", "r") as f:
    text = f.readlines()
    for line in text:
        print(line)





