from OOP_test import Book

if __name__ == '__main__':
    book = Book('Book name', 'Publisher name', '84130139291233',14.99)
    book_two = Book('This is another book name', 'Publisher name', '84130139291232',14.99)
    print(id(book))
    print(id(book_two))

print(Book.__mro__)