import json


def export(list):
    print(len(list))

    with open("negga.json", "w") as file:
        data = {0: list[0], 1: list[1],
                2: list[2], 3: list[3]}
        json.dump(data, file, indent=4)
