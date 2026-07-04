import os
import subprocess
import sys

def run_command(command, desc=None):
    if desc:
        print(f"\n========================================\n{desc}\n========================================")
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        print(f"\nError: Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
    # 1. Install dependencies
    run_command(
        "pip install -r requirements.txt",
        "Step 1: Installing Dependencies"
    )
    
    # 2. Run BC dataset generation
    run_command(
        "python -u scripts/generate_bc_dataset.py",
        "Step 2: Generating Expert Behavior Cloning Dataset"
    )
    
    # 3. Pretrain step policy
    run_command(
        "python -u scripts/train_bc_policy.py --epochs 20 --batch_size 128",
        "Step 3: Pretraining RouteStepPolicy via Behavior Cloning"
    )
    
    # 4. Start RL fine-tuning
    run_command(
        "python -u train.py",
        "Step 4: Running Dreamer RL Fine-Tuning"
    )

if __name__ == '__main__':
    main()
