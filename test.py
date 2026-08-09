import os

book_folder = "book" 

book_name = "The Richest Man in Babylon.pdf"

pdf_path = os.path.join(book_folder, book_name)


print(repr(pdf_path))
print(repr("books/The Richest Man in Babylon.pdf"))