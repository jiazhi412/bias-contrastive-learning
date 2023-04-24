import csv
from sklearn import preprocessing
import numpy as np
import torch

# statistics and bias split income, gender
def split_by_attr(xs, y_labels, a_labels):
    pp, pn, np, nn = [], [], [], []
    for i, (x, y, a) in enumerate(zip(xs, y_labels, a_labels)):
        if a == 1 and y == 1:
            pp.append(i)
        elif a == 1 and y == 0:
            pn.append(i)
        elif a == 0 and y == 1:
            np.append(i)
        elif a == 0 and y == 0:
            nn.append(i)
    return pp, pn, np, nn

# This function takes a column of the raw matrix and finds unique values for encoding
def uniqueItems(column):
    list = np.unique(column)
    fixedList = []
    for item in list:
        fixeListItem = [item]
        fixedList.append(fixeListItem)
    return fixedList

def newWidth(column):
    return len(np.unique(column))

def encodeColumn(oldCol, encoder):
    newCol = []
    for c in oldCol:
        c_array = np.array(c).reshape(-1, 1)
        newCol.append(encoder.transform(c_array).toarray())
    return np.array(newCol)

# Binarize the column with trueVal becoming +1 and everything else -1
def binarizeColumn(oldCol, trueVal):
    return [1 if x==trueVal else -1 for x in oldCol]

def data_processing(path):
    char_indices = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20]
    # char_indices = [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20, 21]
    dicts = get_dict(path)
    # Read the raw data file into arrays
    with open(path) as rawDataFile:
        csvReader = csv.reader(rawDataFile, delimiter=',', quotechar='|')
        rows = []
        for row in csvReader:
            cols = []
            for i in range(len(row)):
                # Change the value here into a floating point number
                cur_dict = dicts[i]
                col = row[i]
                if i in char_indices:
                    value = float(cur_dict[col])
                else:
                    value = float(col)
                cols.append(value)
                i += 1
            rows.append(cols)
    newData = data_transform(rows, char_indices)
    print(newData.shape)
    return newData

def data_transform(rows, char_indices):
    """ Data transformation by column:
        Use One-Hot encoding for the categorical features
        and use Gaussian normalization for numerical features
        (translate feature to have zero mean and unit variance)
    """
    rowCount = len(rows)
    colCount = len(rows[0])
    # read it into an ndarray
    arr = np.array(rows)
    ws = []
    for i in range(21):
        if i in char_indices:
            # the widths of the new columns
            ws.append(newWidth(arr[:,i]))
        else:
            ws.append(1)
    print(ws)
    # Create a placeholder for new data
    newData = np.zeros((rowCount, sum(ws)))
    # populate the matrix with the new columns
    c = 0; # index of current column (relative to old data)
    for i in range(21):
        if i in char_indices:
            enc = preprocessing.OneHotEncoder()  
            enc.fit(uniqueItems(arr[:, i]))
            col = encodeColumn(arr[:, i], enc)
            newData[:,c:c+ws[i]] = col.reshape((rowCount, ws[i])); c=c+ws[i]
        else:
            col = preprocessing.scale(arr[:, i])  # numeric
            newData[:, c] = col;c = c + 1
    return newData

def quick_load(path):
    # Read the raw data file into arrays
    with open(path) as rawDataFile:
        csvReader = csv.reader(rawDataFile, delimiter=',')
        rows = []
        for row in csvReader:
            cols = []
            for col in row:
                # Change the value here into a floating point number
                value = float(col)
                cols.append(value)
            rows.append(cols)

    rowCount = len(rows)
    colCount = len(rows[0])

    # read it into an ndarray
    arr = np.array(rows)
    return arr

def get_dict(path):
    with open(path) as rawDataFile:
        csvReader = csv.reader(rawDataFile, delimiter=',', quotechar='|')
        char_indices = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20]
        # char_indices = [1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20, 21]
        dicts = []
        for i in range(21):
            dicts.append(dict())

        for row in csvReader:
            for i in range(len(row)):
                if i in char_indices:
                    col = row[i]
                    cur_dict = dicts[i]
                    if col not in cur_dict:
                        cur_dict[col] = len(cur_dict) + 1
                i += 1
    return dicts


def get_bias(data, bias_name):
    if bias_name == 'label':
        bias_name = 'credit'
    corresponding_dict = {
        "soeca" : (0,4),
        "duration" : (4,1),
        "credit" : (5,5),
        "purpose" : (10,10),
        "credit_amount" : (20,1),
        "savings_account" : (21,5),
        "present_employment_since" : (26,5),
        "installment_rate" : (31,1),
        "sex" : (32,2),
        "debtors" : (34,3),
        "present_residence_since" : (37,1),
        "property" : (38,4),
        "age" : (42,1),
        "other_installment_plans" : (43,3),
        "housing" : (46, 3),
        "number_of_existing_credits" : (49, 1),
        "job" : (50, 4),
        "nopbltpmaf" : (54, 1), # Number of people being liable to provide maintenance for
        "telephone" : (55, 2),
        "foreign_worker" : (57, 2),
        "credit" : (59, 2)
    }
    address, length = corresponding_dict[bias_name]
    return data[:,address:address+length]

def perturb(data, bias_name):
    if bias_name == 'label':
        bias_name = 'credit'
    corresponding_dict = {
        "soeca" : (0,4),
        "duration" : (4,1),
        "credit" : (5,5),
        "purpose" : (10,10),
        "credit_amount" : (20,1),
        "savings_account" : (21,5),
        "present_employment_since" : (26,5),
        "installment_rate" : (31,1),
        "sex" : (32,2),
        "debtors" : (34,3),
        "present_residence_since" : (37,1),
        "property" : (38,4),
        "age" : (42,1),
        "other_installment_plans" : (43,3),
        "housing" : (46, 3),
        "number_of_existing_credits" : (49, 1),
        "job" : (50, 4),
        "nopbltpmaf" : (54, 1), # Number of people being liable to provide maintenance for
        "telephone" : (55, 2),
        "foreign_worker" : (57, 2),
        "credit" : (59, 2)
    }
    address, length = corresponding_dict[bias_name]
    data[:,address:address+length] = torch.ones_like(data[:,address:address+length]) / length
    return data

if __name__ == '__main__':
    # read data and save data
    load_path = '/nas/vista-ssd01/users/jiazli/datasets/German/GermanData_sex.csv'
    newData = data_processing(load_path)

    # Save to csv file
    save_path = '/nas/vista-ssd01/users/jiazli/datasets/German/German_newData.csv'
    # save_path = 'German_newData.csv'
    np.savetxt(save_path, newData, delimiter=",")

    # read save data directly to save time
    save_path = '/nas/vista-ssd01/users/jiazli/datasets/German/German_newData.csv'
    # save_path = 'German_newData.csv'
    data = quick_load(save_path)
    print(data.shape)

    # test get bias
    bias_data = get_bias(data,"sex")
    print(bias_data.shape)





    # corresponding_dict = {
    #     "soeca" : (0,4),
    #     "duration" : (4,1),
    #     "credit" : (5,5),
    #     "purpose" : (10,10),
    #     "credit_amount" : (20,1),
    #     "savings_account" : (21,5),
    #     "present_employment_since" : (26,5),
    #     "installment_rate" : (31,1),
    #     "personal_status_and_sex" : (32,4),
    #     "debtors" : (36,3),
    #     "present_residence_since" : (39,1),
    #     "property" : (40,4),
    #     "age" : (44,1),
    #     "other_installment_plans" : (45,3),
    #     "housing" : (48, 3),
    #     "number_of_existing_credits" : (51, 1),
    #     "job" : (52, 4),
    #     "nopbltpmaf" : (56, 1), # Number of people being liable to provide maintenance for
    #     "telephone" : (57, 2),
    #     "foreign_worker" : (59, 2),
    #     "credit" : (61, 2)
    # }