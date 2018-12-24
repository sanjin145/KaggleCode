import pandas as pd
import random
import sys


test_arr = []
for i in range(1000):
    single_data = []
    for j in range(783):
        single_data.append(random.randint(0,255))
    test_arr.append(single_data)

    if i % 100 == 0:
        print("构造完成%d条数据" % (i))

print(sys.getsizeof(test_arr)/1024)

columnArr = []
for i in range(783):
    columnArr.append('pixel' + str(i))

test_pd = pd.DataFrame(test_arr,columns=columnArr)
print(sys.getsizeof(test_pd)/(1024*1024))

print(test_pd.shape)
print(type(test_pd))
test_list = test_pd.iloc[:20,].div(255.0)
print(test_list)

#print(test_list)