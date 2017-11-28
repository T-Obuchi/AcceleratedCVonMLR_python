# coding=utf-8


def show_me(variable, symbol_dictionary, length=None):
    for key, value in symbol_dictionary.items():
        if id(variable) == id(value):
            print("## " + key)
            print(" ", variable.shape)
            if not length:
                print(" ", variable)
                print()
            else:
                print(" ", variable[:length])
                print()
            return

    print("not found")
    print()
