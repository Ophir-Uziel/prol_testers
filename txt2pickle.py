import pickle
import ast

# with open('1_name.txt', 'r') as f:
#     mylist = ast.literal_eval(f.read())
#
# print(mylist)
list = []
f = open('surnames english.txt', 'r')
x = f.read()
for h in x.splitlines():
    if h not in list:
        list.append(h)
f.close()


list = list[1:]

print(list)

pickle.dump( list, open("more_last.p", "wb" ) )
