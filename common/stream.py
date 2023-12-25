import numpy as np


def print_matrix(array: np.ndarray) -> None:
    for row in array:
        print(*list(row))


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█',
                      printEnd = "\r", properties={}):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    prop_string = " | ".join([f"{k}: {v}" for k, v in properties.items()])
    print(f'\r{prefix} |{bar}| {percent}% {suffix} | {prop_string}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()