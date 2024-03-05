import subprocess
import concurrent.futures
import time

command = "python cifar10-scaling.py"
num_iterations = 5
delay = 10

def run_command(command):
    subprocess.run(command, shell=True)

for _ in range(8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_iterations) as executor:
        futures = []
        for _ in range(num_iterations):
            time.sleep(delay)
            future = executor.submit(run_command, command)
            futures.append(future)
        concurrent.futures.wait(futures)

    time.sleep(10)
