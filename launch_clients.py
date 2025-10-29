import subprocess
import time

# Number of clients to run
NUM_CLIENTS = 5  # change as needed

# Path to the client script
CLIENT_SCRIPT = "c1.py"

processes = []

for i in range(1, NUM_CLIENTS + 1):
    print(f"[🚀 Launching Client {i}]")
    # Pass client_id as environment variable so each client can load its data
    p = subprocess.Popen(["python", CLIENT_SCRIPT, str(i)])
    processes.append(p)
    time.sleep(1)  # slight delay to avoid race conditions

# Wait for all clients to finish
for p in processes:
    p.wait()

print("[✅] All clients finished execution.")
