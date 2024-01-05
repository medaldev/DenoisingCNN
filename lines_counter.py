import os
import sys
answer = []


def get_info(folder, nesting):
    global answer
    structure = {"folders": 0, "files": 0, "strings": 0}
    for obj in os.scandir(folder):
        if obj.is_dir():
            blacklist = ["__pycache__", "force_of_sharps", "TestExample", "venv", "venv39", "static", "jupyter_notes", "rubbush",
                         "investapi", "soft", "researches", "indicators", "logs", "app", "tradearea",
                         "migrations", "static", "runs", "data"]
            if obj.name.startswith("."):
                continue
            if obj.name in blacklist:
                continue
            children_structure = get_info(obj.path, nesting + 1)
            structure["folders"] += 1
            structure["folders"] += children_structure["folders"]
            structure["files"] += children_structure["files"]
            structure["strings"] += children_structure["strings"]
        elif obj.is_file():
            blacklist = ["__init__.py"]
            if obj.name.startswith("."):
                continue
            if obj.name in blacklist:
                continue
            if obj.name.endswith(".py"):
                structure["files"] += 1
                with open(obj.path, "r", encoding="utf-8") as file:
                    strings = len(file.readlines())
                    structure["strings"] += strings
                answer += ["\t" * (nesting + 1) + f"file {obj.name} {strings} "
                           f"строк"]
    name = folder.split('\\')[-1]
    answer += ["\t" * nesting + f"folder {name} "
               f"{structure['folders']} папок {structure['files']} файлов "
               f"{structure['strings']} строк"]
    return structure


if __name__ == '__main__':
    get_info(os.getcwd(), 0)
    print(*answer[::-1], sep="\n")