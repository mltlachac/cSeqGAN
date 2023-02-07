import pandas as pd
from sklearn.model_selection import train_test_split

dep = pd.read_csv('data/interview_Depressed.csv', names=["text"])
dep['depressed'] = dep.apply(lambda x: 1, axis=1)
ndep = pd.read_csv('data/interview_NotDepressed.csv', names=["text"])
ndep['depressed'] = ndep.apply(lambda x: 0, axis=1)

dep = dep.sample(frac=1, random_state=42).reset_index(drop=True)
ndep = ndep.sample(frac=1, random_state=42).reset_index(drop=True)

#print(dep)
test_dep = dep[0:300]
test_ndep = ndep[0:300]
train_dep = dep[300: 1500]
train_ndep = ndep[300:1500]
#print(test_dep)

test = pd.concat([test_dep, test_ndep])
train = pd.concat([train_dep, train_ndep])

test = test.sample(frac=1, random_state=42).reset_index(drop=True)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

#print(test)

dep[300:].to_csv('data/interview_real_dep_train.csv', index=False, header=False)
ndep[300:].to_csv('data/interview_real_ndep_train.csv', index=False, header=False)

#train.to_csv('data/interview_real_train.csv', index=False, header=False)
#test.to_csv('data/interview_real_test.csv', index=False, header=False)