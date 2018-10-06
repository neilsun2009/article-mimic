filename = 'collection-100.txt'
num = 0
with open(filename) as f:
  line = f.readline()
  while line:
    if len(line) <= 1: # discard empty line
      line = f.readline()
      continue
    num += 1
    foutput = open("collection-100-%d" % num, 'w')
    print(line, file=foutput)
    foutput.close()
    line = f.readline()