import pandas as pd

from model import predict

print("Welcome to the credit application prediction model, go ahead and input your data after this and it will return a decision for you")
gender = input("Input 0 for female and 1 for male: ")
age = input("Input the age: ")
married = input("Input 1 if married and 0 if other: ")
customer = input("Input 0 if not a bank customer, input 1 if bank customer: ")
ethnicity = input("Input ethnicity as 0 for white, 1 for black, 2 for asian, 3 for latino, 4 for other: ")
years_employed = input("Input years employed: ")
prior_default = input("Input 0 if there is no prior default, 1 if there is a prior default: ")
employed = input("Input 0 if not employed, 1 if employed: ")
driver =  input("Input 0 if not if drivers license is not held, input 1 if it is: ")
#
input_features = {
    'Gender': gender,
    'Age': age,
    'Married': married,
    'BankCustomer': customer,
    'Ethnicity': ethnicity,
    'YearsEmployed': years_employed,
    'PriorDefault': prior_default,
    'Employed': employed,
    'DriversLicense': driver
}

input_df = pd.DataFrame([input_features])
decision = predict(input_df)[0]
if decision == 0:
    print("\nThe decision is Denied")
if decision == 1:
    print("\nThe decision is Approved")