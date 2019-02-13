#importing libraries
import pandas
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import math
import random
import textwrap
import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#importing dataset
df =  pandas.read_excel('dataset.xlsx')
df = df.loc[:,['Weight lbs','Height inch', 'Neck circumference', 'Chest circumference', 'Abdomen  circumference', 'bodyfat']]

#spliting training and test data
train, test, = train_test_split(df, test_size=0.25)

#Normalizing training and test data
normalized_train = preprocessing.normalize(train)
normalized_test = preprocessing.normalize(test)


#trimming body column from the dataset
y_train = normalized_train[:,0:5]
y_test = normalized_test[:,0:5] 

#print(len(y_test))

#creating weights to achieve yhat values with 500 'W' matrix
weights_list = np.random.uniform(-1,1,(500,5,10))


yhat = 0
temp = 0
temp2 = 0
temp_mat = np.random.uniform(-1,-1,(500,189))

#creating yhat values with 500*189 matrix
for i in range(len(weights_list)):
    for j in range(len(y_train)):
        temp = np.dot(y_train[j],weights_list[i]) 
        for k in range(len(temp)):
            temp2 = 1 / (1 + math.exp(-temp[k]))
            yhat = yhat + temp2
            temp2 = 0
        temp_mat[i][j] = yhat
        temp = 0
        yhat = 0

               
fitness_value = [None]*500
square = 0
for i in range(len(weights_list)):
    for j in range(len(y_train)):
        temp = np.power((temp_mat[i][j] - y_train[j][4]), 2)
        square = square + temp
        temp = 0
    fitness_value[i] = ((1 - (square/189)) * 100)
    square = 0

#print(fitness_value) 
    
#Parent value and index 
max_fitness_value = []
parent = (max(fitness_value))
parent_index = (fitness_value.index(max(fitness_value)))
max_fitness_value.append(parent)
print("Fitness Value (before iteration): "+str(parent))

#calculating binary matrix
binary_matrix = ['a']*500
chromozome = []
for i in range(len(weights_list)):
    binary_matrix[i] = ((weights_list[i] - np.min(weights_list[i]))/np.ptp(weights_list[i]) * 1000).astype(int)
    
temp_list = []
temp = ''
for i in range(len(binary_matrix)):
    for j in range(len(binary_matrix[i])):
        for k in range(len(binary_matrix[i][j])):
            temp = (bin(binary_matrix[i][j][k])[2:].zfill(10))
            temp_list.append(temp)
            #print(len(temp_list))
            
temp = ''
temp_list = np.reshape(temp_list, (500,5,10))
         

#calculating chromozome
for i in range(len(temp_list)):
    temp = ''
    for j in range(len(temp_list[i])):
        for k in range(len(temp_list[i][j])):
            temp = temp + temp_list[i][j][k]
    chromozome.append(temp)
    
    
#Cross Over of chromosome
rand = 0
offsprings = []
parent1 = ''
parent2 = ''
child1 = ''
child2 = ''
for i in range(0,500):
    rand = random.randint(1,500)
    parent1 = chromozome[parent_index]
    parent2 = chromozome[i]
    child1 = (parent1[:rand] + parent2[rand:])
    child2 = (parent1[rand:] + parent2[:rand])
    offsprings.append(child1)
    offsprings.append(child2)


#Constructing 2*500 mutation values
mutation = []  
temp = ''
for i in range(len(offsprings)):
    rand = 0
    temp = (offsprings[i])
    for j in range(25):
        rand = random.randint(0,499)
        if(temp[rand] is '1'):
            temp = temp[:rand]+'0'+temp[rand+1:]
        else:
            temp = temp[:rand]+'1'+temp[rand+1:]
    
    mutation.append(temp)


    
#debinarization and denormalization
debinary1 = []
debinary2 = []
temp = ''
temp2 = 0
weights_list2 = []
child_denormalized = []
for i in range(len(mutation)):
        debinary1.append(textwrap.wrap(mutation[i], 10))
        for j in range(0,50):
            temp = debinary1[i][j]
            decimal_x = int(temp,2)
            temp2 = float(decimal_x)/1000
            debinary1[i][j] = temp2
            child_denormalized.append((temp2*4)-2)

weights_list2 = np.reshape(child_denormalized, (1000,5,10))

temp_mat2 = np.random.uniform(-1,-1,(1000,189))

for i in range(len(weights_list2)):
    for j in range(len(y_train)):
        temp = np.dot(y_train[j],weights_list2[i]) 
        for k in range(len(temp)):
            temp2 = 1 / (1 + math.exp(-temp[k]))
            yhat = yhat + temp2
            temp2 = 0
        temp_mat2[i][j] = yhat
        temp = 0
        yhat = 0


#calculating a new fitness value
fitness_value2 = [None]*1000
square = 0
for i in range(len(weights_list2)):
    for j in range(len(y_train)):
        temp = np.power((temp_mat2[i][j] - y_train[j][4]), 2)
        square = square + temp
        temp = 0
    fitness_value2[i] = ((1 - (square/189)) * 100)
    square = 0


max_fitness_value.append(max(fitness_value2))    
print("Fitness Value (before iteration): "+str(max(fitness_value2)))
test = max(fitness_value2)
final_fitness_value = []
median_value = statistics.median(fitness_value2)
index_list = []


for i in range(1000):
    if(fitness_value2[i] >= median_value):
        index_list.append(i)
        final_fitness_value.append(fitness_value2)


for i in range(500):
    weights_list[i] = weights_list2[index_list[i]]
    
fitness_value = final_fitness_value
    
##########End of First cycle


#Looping to reach plateau value
for i in range(0,20):
    for i in range(len(weights_list)):
        for j in range(len(y_train)):
            temp = np.dot(y_train[j],weights_list[i]) 
            for k in range(len(temp)):
                temp2 = 1 / (1 + math.exp(-temp[k]))
                yhat = yhat + temp2
                temp2 = 0
            temp_mat[i][j] = yhat
            temp = 0
            yhat = 0


    #Parent value and index
    parent = (max(fitness_value))
    parent_index = (fitness_value.index(max(fitness_value)))

    #calculating binary matrix
    binary_matrix = ['a']*500
    chromozome = []
    for i in range(len(weights_list)):
        binary_matrix[i] = ((weights_list[i] - np.min(weights_list[i]))/np.ptp(weights_list[i]) * 1000).astype(int)
        
    temp_list = []
    temp = ''
    for i in range(len(binary_matrix)):
        for j in range(len(binary_matrix[i])):
            for k in range(len(binary_matrix[i][j])):
                temp = (bin(binary_matrix[i][j][k])[2:].zfill(10))
                temp_list.append(temp)
                #print(len(temp_list))
                
    temp = ''
    temp_list = np.reshape(temp_list, (500,5,10))
    
    #calculating chromozome
    for i in range(len(temp_list)):
        temp = ''
        for j in range(len(temp_list[i])):
            for k in range(len(temp_list[i][j])):
                temp = temp + temp_list[i][j][k]
        chromozome.append(temp)

    
    #Constructing 2*500 mutation values
    mutation = []  
    temp = ''
    for i in range(len(offsprings)):
        rand = 0
        temp = (offsprings[i])
        for j in range(25):
            rand = random.randint(0,499)
            if(temp[rand] is '1'):
                temp = temp[:rand]+'0'+temp[rand+1:]
            else:
                temp = temp[:rand]+'1'+temp[rand+1:]
        
        mutation.append(temp)
        
    #debinarization and denormalization
    debinary1 = []
    debinary2 = []
    temp = ''
    temp2 = 0
    weights_list2 = []
    child_denormalized = []
    for i in range(len(mutation)):
            debinary1.append(textwrap.wrap(mutation[i], 10))
            for j in range(0,50):
                temp = debinary1[i][j]
                decimal_x = int(temp,2)
                temp2 = float(decimal_x)/1000
                debinary1[i][j] = temp2
                child_denormalized.append((temp2*3.6)-2.7)
    
    weights_list2 = np.reshape(child_denormalized, (1000,5,10))
    
    temp_mat2 = np.random.uniform(-1,-1,(1000,189))
    
    for i in range(len(weights_list2)):
        for j in range(len(y_train)):
            temp = np.dot(y_train[j],weights_list2[i]) 
            for k in range(len(temp)):
                temp2 = 1 / (1 + math.exp(-temp[k]))
                yhat = yhat + temp2
                temp2 = 0
            temp_mat2[i][j] = yhat
            temp = 0
            yhat = 0
    
    
    fitness_value2 = [None]*1000
    square = 0
    for i in range(len(weights_list2)):
        for j in range(len(y_train)):
            temp = np.power((temp_mat2[i][j] - y_train[j][4]), 2)
            square = square + temp
            temp = 0
        fitness_value2[i] = ((1 - (square/189)) * 100)
        square = 0
    
    
    final_fitness_value = []
    median_value = statistics.median(fitness_value2)
    index_list = []
    #print(median_value)
    for i in range(1000):
        if(fitness_value2[i] >= median_value):
            index_list.append(i)
            final_fitness_value.append(fitness_value2)
            
    #checking if new parent is greater than previous parent  
    max_pos = 0 #to have the index of highest fitness value    
    if(test < max(fitness_value2)):
        max_pos = fitness_value2.index(max(fitness_value2))
        print("Fitness Value (After Iteration): "+str(max(fitness_value2)))
        test = max(fitness_value2)
        for i in range(500):
            weights_list[i] = weights_list2[index_list[i]]
        fitness_value = final_fitness_value
    
    max_fitness_value.append(max(fitness_value2))


#scatter plot for fitness value for iteration   
x = np.array(range(0,20))
y = np.array(max_fitness_value[2:])
plt.scatter(x,y,alpha=0.5)
plt.title("Fitness value for iteration")
plt.ylabel("Fitness")
plt.xlabel("No of iteration")
plt.show()


#Error calculation

#creating yhat values with 500*63 matrix
for i in range(len(weights_list)):
    for j in range(len(y_test)):
        temp = np.dot(y_test[j],weights_list[i]) 
        for k in range(len(temp)):
            temp2 = 1 / (1 + math.exp(-temp[k]))
            yhat = yhat + temp2
            temp2 = 0
        temp_mat[i][j] = yhat
        temp = 0
        yhat = 0

#calculating an error
error = []
square = 0
test_col1 = []
test_col2 = []
test_y = []
test_yhat = []
#Calculating Overall Error Value
for i in range(len(y_test)):
    temp = np.power((temp_mat[max_pos][i] - y_test[i][4]), 2)
    square = square + temp
    test_col1.append(y_test[i][0])
    test_col2.append(y_test[i][1])
    test_y.append(y_test[i][4])
    test_yhat.append(temp_mat[max_pos][i])
print("Overall Error is: "+str(square/len(y_test)))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = test_col1
y = test_col2
z = test_yhat
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Weight')
ax.set_ylabel('Height')
ax.set_zlabel('Yhat')
ax.set_title('Fitness value for Yhat values')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = test_col1
y = test_col2
z = test_y
ax.scatter(x, y, z, c='b', marker='^')

ax.set_xlabel('Weight')
ax.set_ylabel('Height')
ax.set_zlabel('Y')
ax.set_title('Fitness value for Y values')

plt.show()

