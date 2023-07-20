
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
       print("=-=-=-=- END OF OUTPUT -=-=-=-=")
    elif result:
        if len(result.elements) == 1:
            print("")
            print(repr(result.elements[0]))
            print("=-=-=-=- END OF OUTPUT -=-=-=-=")
        else:
            print("")
            print(repr(result))
            print("=-=-=-=- END OF OUTPUT -=-=-=-=")