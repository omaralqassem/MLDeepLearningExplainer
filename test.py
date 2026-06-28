import pandas as pd

data = pd.read_csv('diabetes.csv')



# X = data.drop(columns=['Outcome'])
# y = data['Outcome']
# print(y)

print(data.iloc[4])