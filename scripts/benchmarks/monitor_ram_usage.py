import subprocess
import psutil
import time
import sys

if len(sys.argv) < 2:
    print("Usage: python memory_monitor.py script.py")
    sys.exit(1)

script_path = sys.argv[1]
script_args = sys.argv[2:]  # Pass through any additional arguments

print(f"Monitoring: {script_path}")
print("-" * 60)

# Start the subprocess
cmd = [sys.executable, script_path] + script_args
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Monitor the process
ps_process = psutil.Process(process.pid)
peak_rss = 0
peak_vms = 0
start_time = time.time()

while process.poll() is None:
    try:
        mem_info = ps_process.memory_info()
        rss_gb = mem_info.rss / 1e9  # Physical RAM
        vms_gb = mem_info.vms / 1e9  # Total (RAM + Swap)
        
        peak_rss = max(peak_rss, rss_gb)
        peak_vms = max(peak_vms, vms_gb)
        
        swap_gb = vms_gb - rss_gb
        print(f"RAM: {rss_gb:6.2f}GB | Swap: {swap_gb:6.2f}GB | Total: {vms_gb:6.2f}GB", end='\r')
        
        time.sleep(0.5)
    except psutil.NoSuchProcess:
        break

# Get script output
stdout, stderr = process.communicate()
end_time = time.time()

# Print results
print("\n" + "-" * 60)
if stdout:
    print(stdout)
if stderr:
    print(stderr)

print("-" * 60)
print("MEMORY USAGE SUMMARY")
print("-" * 60)
print(f"Execution time: {end_time - start_time:.1f}s")
print(f"Peak RAM: {peak_rss:.2f}GB")
print(f"Peak Swap: {peak_vms - peak_rss:.2f}GB")
print(f"Peak Total: {peak_vms:.2f}GB")
print("-" * 60)

sys.exit(process.returncode)
