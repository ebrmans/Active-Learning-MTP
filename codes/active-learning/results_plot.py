from plot_extrapolation_grade_hist import plot_extrapolation_grade_hist
import os
import numpy as np

os.makedirs('plots', exist_ok=True)

def calc_max_extrapolation_grade():
    # Fix for not calculating max extrapolation grade in early iterations
    for i in range(61):
        with open(f'iter/{i}/mv_hist.txt','r') as f:
            lines = f.readlines()
        if (len(lines)<3):
            print(f'Iteration {i}: calculating max extrapolation grade')
            # read preselected configuration to get the max extrapolation grade
            mean_mv, std_mv, max_mv = plot_extrapolation_grade_hist(log_file=f'iter/{i}/lammps.log', savedir=f'iter/{i}')
            lines.append(f'max: {max_mv}\n')
            with open(f'iter/{i}/mv_hist.txt','w') as f:
                f.writelines(lines)
    pass

def plot_max_extrapolation_grade_over_iterations(max_iter=60):
    import matplotlib.pyplot as plt

    iterations = []
    max_mvs = []

    for i in range(max_iter+1):
        with open(f'iter/{i}/mv_hist.txt','r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('max:'):
                max_mv = float(line.split()[1])
                iterations.append(i)
                max_mvs.append(max_mv)
                break

    plt.plot(iterations, max_mvs, marker='o')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Max Extrapolation Grade')
    plt.title('Max Extrapolation Grade over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/max_extrapolation_grade_over_iterations.pdf')
    plt.close()

def plot_number_vasp_runs_over_iterations(max_iter=60):
    import matplotlib.pyplot as plt

    iterations = []
    num_vasp_runs = []

    for i in range(max_iter+1):
        vasp_dir = f'iter/{i}/vasp_threshold'
        if os.path.exists(vasp_dir):
            num_runs = len(os.listdir(vasp_dir))
        else:
            num_runs = 0
        iterations.append(i)
        num_vasp_runs.append(num_runs)

    plt.plot(iterations, num_vasp_runs, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Number of VASP Runs')
    plt.title(f'Total number of runs: {np.sum(num_vasp_runs)}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/number_of_vasp_runs_over_iterations.pdf')
    plt.close()

def plot_mean_stddev_extrapolation_grade_over_iterations(max_iter=60):
    import matplotlib.pyplot as plt

    iterations = []
    mean_mvs = []
    stddev_mvs = []

    for i in range(max_iter+1):
        with open(f'iter/{i}/mv_hist.txt','r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('mean:'):
                mean_mv = float(line.split()[1])
            if line.startswith('std:'):
                stddev_mv = float(line.split()[1])
        iterations.append(i)
        mean_mvs.append(mean_mv)
        stddev_mvs.append(stddev_mv)

    plt.errorbar(iterations, mean_mvs, yerr=stddev_mvs, fmt='o', ecolor='r', capthick=2)
    # plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Extrapolation Grade ± Std Dev')
    plt.title('Mean Extrapolation Grade over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/mean_stddev_extrapolation_grade_over_iterations.pdf')
    plt.close()

def plot_sampled_mean_error_energy_forces(max_iter=60):
    import matplotlib.pyplot as plt

    iterations = []
    energy_error = []
    forces_error = []

    for i in range(max_iter+1):
        with open(f'iter/{i}/sampled_mean_error.txt','r') as f:
            lines = f.readlines()

        iterations.append(i)
        energy_error.append(float(lines[0].split()[-1]))
        forces_error.append(float(lines[1].split()[-1]))
        
    # plt.errorbar(iterations, mean_mvs, yerr=stddev_mvs, fmt='o', ecolor='r', capthick=2)
    # plt.yscale('log')
    plt.figure()
    plt.plot(iterations, energy_error)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Energy error')
    # plt.title('Mean Extrapolation Grade over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/energy_error_over_iterations.pdf')
    plt.close()

    plt.figure()
    plt.plot(iterations, forces_error)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Forces error')
    # plt.title('Mean Extrapolation Grade over Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/forces_error_over_iterations.pdf')
    plt.close()

    pass

if __name__ == "__main__":
    # calc_max_extrapolation_grade()
    # plot_max_extrapolation_grade_over_iterations(max_iter=60)
    # plot_number_vasp_runs_over_iterations(max_iter=60)
    # plot_mean_stddev_extrapolation_grade_over_iterations(max_iter=60)
    plot_sampled_mean_error_energy_forces(max_iter=60)
    pass