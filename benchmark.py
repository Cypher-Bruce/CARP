import subprocess
import os
import re
import numpy as np
import csv

def run_benchmark():
    path_to_dataset = os.path.join(os.path.dirname(__file__), 'CARP_datasets')
    csv_file_name = 'benchmark_results2.csv'
    print('Running benchmark on:', path_to_dataset)
    files = os.listdir(path_to_dataset)
    if len(files) == 0:
        print('No dataset found')
        return
    print('Find datasets:')
    for file in files:
        print('\t', file)
        
    with open(csv_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Dataset', 'Mean score', 'Std deviation', 'Best score', 'Raw scores'])
        print(f"Writing results to {csv_file_name}")

        for file in files:
            print(f"Running dataset: {file}")

            full_file_path = os.path.join(path_to_dataset, file)
            random_seeds = [1, 2, 3, 4, 5]
            command = ["python", "CARP_solver.py", full_file_path, "-t", "60", "-s", ""]
            scores = []

            exception = False
            for seed in random_seeds:
                command[-1] = str(seed)
                process = subprocess.run(command, capture_output=True, text=True)
                if process.returncode == 0:
                    match = re.search(r"q\s+([\d.]+)", process.stdout)
                    if match:
                        scores.append(int(match.group(1)))
                    else:
                        print("Error: Cannot parse the output", process.stdout)
                        exception = True
                        break
                else:
                    print("Error: Cannot run the command", command)
                    print("Error message:", process.stderr)
                    exception = True
                    break
                
            if exception:
                print(f"Dataset {file} is skipped due to exception")
                continue

            mean_score = np.mean(scores)
            std_dev_score = np.std(scores)
            best_score = np.min(scores)

            print(f"Dataset: {file}")
            print(f"\tMean score: {mean_score}")
            print(f"\tStd deviation: {std_dev_score}")
            print(f"\tBest score: {best_score}")
            print(f"\tRaw scores: {scores}")
            writer.writerow([file, mean_score, std_dev_score, best_score, scores])

if __name__ == '__main__':
   run_benchmark()