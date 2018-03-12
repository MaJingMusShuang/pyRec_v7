file = open('trusts.txt')
lines = []
for line in file:
    user1, user2 =line.split(' ')
    user1 = int(user1)
    user2 = int(user2)
    writeLine = '{} {} 1\n'.format(user1, user2)
    lines.append(writeLine)
file.close()
with open('trusts.txt', 'w') as writeFile:
    writeFile.writelines(lines)