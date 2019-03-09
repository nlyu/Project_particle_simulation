import re
import os

# Parse output file
tmpfile = open('auto-particle-gpu.stdout','r')
m = re.findall("n = (.*), simulation time = (.*) seconds",tmpfile.read())
# Generate tmp file  to be input for autograder  
f= open("out.tmp","w") 
for i in m:
	f.write(i[0] + " " + i[1] + "\n")
	print(i[0] + " " + i[1])

f.close();

# Run autograder
os.system("./autograder -v gpu -s out.tmp ")
