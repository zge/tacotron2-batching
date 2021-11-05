import csv

def dict2row(dct, csvname='filename.csv', delimiter=',', order=None, verbose=True):
    keys = list(dct.keys())
    if order == 'ascend':
        keys = sorted(keys)
    elif order == 'descend':
        keys = sorted(keys, reverse=True)
    with open(csvname, 'w', newline='') as f:
        csv_out = csv.writer(f, delimiter=delimiter)
        for k in keys:
          row = [k, str(dct[k])]
          csv_out.writerow(row)
    if verbose:
        print('{} saved!'.format(csvname))
