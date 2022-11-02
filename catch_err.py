def do_smth():
    print("hi")
    open("sf.txt")
    raise ValueError("Do not want to")


try:
    do_smth()
except ValueError as v:
    print(v, "but does not matter")
except FileNotFoundError as f:
    print(f)
