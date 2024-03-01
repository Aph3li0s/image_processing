from collections import Counter
buffer = [0] * 5
import random
for i in range(10):
    n = random.randint(1,3)
    if n == 1:
        print(1)
        buffer.append(1)
    if n == 2:
        buffer.append(2)
    if n == 3:
        buffer.append(3)    
    buffer.pop(0)
print(buffer)
from collections import Counter
a = [element for element, count in Counter(buffer).items() if count == max(Counter(buffer).values())]
if len(a) != 1:
    print(a)