

###Calling it from command-line
from subprocess import Popen, PIPE
p = Popen(['../run.sh', 'GenericBatchUser', 'sample.txt'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate(b"input data that is passed to subprocess' stdin")




