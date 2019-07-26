class Dog:

    # Class Attribute
    species = 'mammal'

    # Initializer / Instance Attributes
    def __init__(self, name, age):
        self.name = name
        self.age = age

Ruff = Dog("Ruff",10)
Filo = Dog("Filo", 7)
Martin = Dog("Martin",2)

def get_biggest_number(*args):
    return max(args)

print("The oldest dog is {} years old.".format(
    get_biggest_number(Ruff.age, Filo.age, Martin.age)))
