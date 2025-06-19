from subprocess import Popen
base_command = 'python3 parallel_main.py --proc_id '

# procs = [ Popen(i) for i in commands ]
n = 4
procs = []
for i in range(n):
    command_str = base_command+str(i)
    procs.append(Popen(command_str,shell=True))

for p in procs:
   p.wait()
