
text = "Welcome to Strive"
new_text = text.replace("e", "i")
print(new_text)

text1 = "Welcome to Strive           "
print([text1.strip()])
print(text1.startswith("i"))


new_text1 = text1.split()
print(new_text1)

new_text2 = text1.split("to")
print(new_text2)

email = "andrea.viviani89@gmail.com"

new_email = email.split("@")
print(new_email[0])