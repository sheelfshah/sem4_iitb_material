import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""

def normalize(df):
    global phase, means, ranges
    df = np.array(df, "float64")
    if phase=="train":
        means = np.mean(df, axis=0)
        ranges = np.ptp(df, axis=0)
        zeros = np.where(ranges==0)
        ranges[zeros] = 1
        means[zeros] = 0
        return (df-means)/ranges
    return (df-means)/ranges


def one_hot(df, col_name, vals):
    if len(vals)==2:
        val = vals[0]
        df[col_name + "_binary"] = df[col_name].apply(lambda x: 1 if x==val else 0)
        return df.drop(col_name, axis=1)
    for val in vals:
        df[col_name + "_" + val] = df[col_name].apply(lambda x: 1 if x==val else 0)
    return df.drop(col_name, axis=1)

def preprocess(df):
    df["bias"] = 1

    df["company"] = df["name"].apply(lambda x : x.split()[0])

    fuel_types = ["Diesel", "Petrol", "CNG", "LPG"]
    seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
    transmission_types = ['Manual', 'Automatic']
    owner_types = ['First Owner','Second Owner', 'Third Owner',
    'Fourth & Above Owner', 'Test Drive Car']
    company_types = ['Maruti', 'Skoda', 'Honda', 'Hyundai',
    'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata',
    'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz',
    'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan',
    'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
    'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel']

    col_and_type_dict = {
    "fuel": fuel_types,
    "seller_type": seller_types,
    "transmission": transmission_types,
    "owner": owner_types,
    "company": company_types
    }

    for col_name in col_and_type_dict.keys():
        df = one_hot(df, col_name, col_and_type_dict[col_name])

    df["mileage"].replace(np.nan, "0 kmpl", regex=True, inplace=True)
    df["mileage"] = df["mileage"].apply(lambda x: float(x.split()[0]))
    df["engine"].replace(np.nan, "0 kmpl", regex=True, inplace=True)
    df["engine"] = df["engine"].apply(lambda x: float(x.split()[0]))

    df.replace(np.nan, 0, inplace=True)

    df.drop(["Index", "name", "torque"], axis=1, inplace=True)

    return normalize(df)

def preprocess_basis(df):
    df["bias"] = 1

    df["company"] = df["name"].apply(lambda x : x.split()[0])

    fuel_types = ["Diesel", "Petrol", "CNG", "LPG"]
    seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
    transmission_types = ['Manual', 'Automatic']
    owner_types = ['First Owner','Second Owner', 'Third Owner',
    'Fourth & Above Owner', 'Test Drive Car']
    company_types = ['Maruti', 'Skoda', 'Honda', 'Hyundai',
    'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata',
    'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz',
    'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan',
    'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
    'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel']

    col_and_type_dict = {
    "fuel": fuel_types,
    "seller_type": seller_types,
    "transmission": transmission_types,
    "owner": owner_types,
    "company": company_types
    }

    for col_name in col_and_type_dict.keys():
        df = one_hot(df, col_name, col_and_type_dict[col_name])

    df["mileage"].replace(np.nan, "0 kmpl", regex=True, inplace=True)
    df["mileage"] = df["mileage"].apply(lambda x: float(x.split()[0]))
    df["engine"].replace(np.nan, "0 kmpl", regex=True, inplace=True)
    df["engine"] = df["engine"].apply(lambda x: float(x.split()[0]))
    df["year"] = df["year"].apply(lambda x: np.log(x))
    df["km_driven"] = df["km_driven"].apply(lambda x: np.log(x))
    df["seats"] = df["seats"].apply(lambda x: np.exp(x))
    df.replace(np.nan, 0, inplace=True)

    df.drop(["Index", "name", "torque"], axis=1, inplace=True)
    

    return normalize(df)

def get_features(file_path):
    # Given a file path , return feature matrix and target labels
    global phase
    df = pd.read_csv(file_path)
    if phase != "test":
        phi = preprocess(df.drop("selling_price", axis=1))
        y = df["selling_price"].to_numpy()
        return phi, y
    phi = preprocess(df)
    return phi, None

def get_features_basis(file_path):
    # Given a file path , return feature matrix and target labels 
    global phase
    df = pd.read_csv(file_path)
    if phase != "test":
        phi = preprocess_basis(df.drop("selling_price", axis=1))
        y = df["selling_price"].to_numpy()
        return phi, y
    phi = preprocess_basis(df)
    return phi, None

def compute_RMSE(phi, w , y) :
    # Root Mean Squared Error
    diff = phi@w - y
    error = (diff*diff).sum()/len(diff)
    error = np.sqrt(error)
    return error

def generate_output(phi_test, w):
    df = pd.DataFrame(phi_test@w)
    df[0] = df[0].apply(lambda x: max(-x, x))
    df.to_csv("output.csv")
    
def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return np.linalg.pinv(phi).dot(y)
    
def gradient_descent(phi, y, phi_dev, y_dev) :
    # Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    w = np.random.normal(0, 0.1, phi.shape[1])
    epochs=int(1e5) # max_num_of_epochs
    lr = 2e-4
    prev_val_rmse = compute_RMSE(phi_dev, w, y_dev)
    ch=[]
    for ep in range(epochs):
        err = (phi.T@(phi@w - y))
        w = w - lr * err
        val_rmse = compute_RMSE(phi_dev, w, y_dev)
        if prev_val_rmse < val_rmse:
            break
        prev_val_rmse = val_rmse
        # ch.append(val_rmse)
    # print(ep)
    # plt.plot(ch[300:])
    # plt.show()
    return w

def sgd(phi, y, phi_dev, y_dev) :
    # Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    w = np.random.normal(0, 0.1, phi.shape[1])
    epochs=int(1e5) # max_num_of_epochs
    lr = 1e-4
    prev_val_rmse = compute_RMSE(phi_dev, w, y_dev)
    ch=[]
    perm = np.random.RandomState(seed=0).permutation(len(phi))
    batch_size=128
    for ep in range(epochs):
        phi_, y_ = phi[perm], y[perm]
        for batch_i in range(len(phi)//batch_size):
            low, high = batch_i*batch_size, (batch_i+1)*batch_size
            err = (phi_[low:high].T@(phi_[low:high]@w - y_[low:high]))
            w = w - lr * err
        val_rmse = compute_RMSE(phi_dev, w, y_dev)
        if prev_val_rmse < val_rmse:
            break
        prev_val_rmse = val_rmse
    #     ch.append(val_rmse)
    # print(ep)
    # plt.plot(ch[300:])
    # plt.show()
    return w


def pnorm(phi, y, phi_dev, y_dev, p) :
    # Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    w = np.random.normal(0, 0.1, phi.shape[1])
    epochs=int(1e5) # max_num_of_epochs
    lr = 1e-4
    lambda_reg2 = 2e-1
    lambda_reg4 = 3e-17
    prev_val_rmse = compute_RMSE(phi_dev, w, y_dev)
    ch=[]
    for ep in range(epochs):
        if p==2:
            err = (phi.T@(phi@w - y)) +  p * w**(p-2) * lambda_reg2 * w
        else:
            err = (phi.T@(phi@w - y)) +  p * w**(p-2) * lambda_reg4 * w
        w = w - lr * err
        val_rmse = compute_RMSE(phi_dev, w, y_dev)
        if prev_val_rmse < val_rmse:
            break
        prev_val_rmse = val_rmse
    return w   

def plot(phi, y, phi_dev, y_dev):
    sizes =  [2000, 2500, 3000, len(phi)]
    rmses=[]
    for size in sizes:
        phi_ = phi[:size]
        y_ = y[:size]
        w = pnorm(phi_, y_, phi_dev, y_dev, 2)
        rmses.append(compute_RMSE(phi_dev, w, y_dev))
    plt.plot(sizes, rmses)
    plt.xlabel(xlabel="Training Size")
    plt.ylabel(ylabel="Validation RMSE")
    plt.savefig("rmse_vs_size.png")
    plt.show()


def main():
    """
    The following steps will be run in sequence by the autograder.
    """
    global phase
    ######## Task 1 #########
    phase = "train"
    phi, y = get_features('train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features('dev.csv')
    plot(phi, y, phi_dev, y_dev)
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))
    print(r1, r2, r3)

    ######## Task 2 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)

    ######## Task 3 #########
    phase = "train"
    phi_basis, y = get_features_basis('train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features_basis('dev.csv')
    phase = "test"
    phi_test, _ = get_features_basis('test.csv')
    w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
    rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
    print('Task 3: basis')
    print(rmse_basis)

    # Task 6
    generate_output(phi_test, w_basis)

main()
