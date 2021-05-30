import subprocess as sp
import time

cmd = ["python", "-m", "pip", "install", "-i", "https://test.pypi.org/simple/", "-r", "version.txt"]
for i in range(10):
    print(" ".join(cmd))
    a = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
    print(a.stdout)
    time.sleep(30)
    if a.returncode == 0:
        print("Success!")
        break
