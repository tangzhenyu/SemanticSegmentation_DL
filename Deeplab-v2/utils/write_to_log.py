def write_log(str, filename):
    with open(filename, 'a') as f:
        f.write(str + "\n")
