import random

lines = []
counter = 0
while True:
    user1 = random.randint(a=1, b=7375)
    user2 = random.randint(a=1, b=7375)
    if user1==user2:
        continue
    lines.append('{} {} 1\n'.format(user1, user2))
    counter += 1
    if counter>=111781:
        break
with open('randomTrusts.txt', 'w') as writeFile:
    writeFile.writelines(lines)