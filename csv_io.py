import csv

# write to csv
def write2csv(filename, data):
    print(filename, "Appending")
    with open(filename, 'a+', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(data)
    print("Write done")
    f.close

# read from csv
def read_from_csv(filename):
    ipds = []
    with open(filename)as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for rows in f_csv:
            ipd = float(rows[0])
            ipds.append(ipd)
    return ipds

