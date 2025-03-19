# # Enumerate
# list1 = ['apple', 'banana', 'cherry']
# for index, value in enumerate(list1):
#     print(index, value)

# print()

# # Count
# list2 = ['apple', 'banana', 'cherry', 'apple']
# print(list2.count('apple'))

# print()

# num = 4
# for i in range(1,10,2):
#     print(" "*num + i*"*")
#     num -= 1

# print()

# name = ['apple', 'banana', 'cherry']
# pay = (50, 30, 20)
# s = ("male", "female", "male")

# newName = []
# for index, sex in enumerate(s):
#     if sex == "female":
#         newName.append(name[index])
# print(newName)

# print()

with open(r'.\data.txt','r') as f:
    data = f.read()
print(data)