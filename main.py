
import Ophidian


def execute(code):
    
    print("#################################")
    print("=-=-=-=- START OF OUTPUT -=-=-=-=")
    print("")
    text = code

    # ignores empty inputs
    if text.strip() == "": return None
    result, error = Ophidian.run('<stdin>', text)

    if error:
       print("")
       print(error.as_string())

    elif result:
        if len(result.elements) == 1:
            print("")
            print(repr(result.elements[0]))

        else:
            print("")
            print(repr(result))
