a=input("input something")
try:
    b = int(a)
except ValueError:
    try:
        b = float(a)
    except ValueError:
        print("a is a string")
    else:
        print("a is a float")
else:
    print("a is an int")