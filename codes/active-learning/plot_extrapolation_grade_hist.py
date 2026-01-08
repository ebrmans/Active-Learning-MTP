import numpy as np
import matplotlib.pyplot as plt

def plot_extrapolation_grade_hist(log_file='lmp_mpi.log', savedir='./'):
    # note that LAMNPS has been modified to output the MV grade 

    with open(log_file,'r') as f:
        lines = f.readlines()

    mv_grade = []

    for line in lines:
        if "MV-grade from MPI_Allreduce is" in line:
            mv_grade.append(float(line.split()[-1]))
            continue

        if "threshold_break" in line:
            threshold_break = float(line.split()[-1])
            continue

        if "threshold_save" in line:
            threshold_save = float(line.split()[-1])
            continue

    mean_mv = np.mean(mv_grade)
    std_mv = np.std(mv_grade)
    max_mv = np.max(mv_grade)

    plt.hist(mv_grade, bins=50)
    plt.title(f'{mean_mv:1.3e} +/- {std_mv:1.3e}')
    plt.yscale('log')
    plt.ylabel(f'Total number: {len(mv_grade)}')
    plt.xlabel(f'Extrapolation grade [-]')
    plt.tight_layout()
    plt.savefig(f'{savedir}/hist_mv.pdf')
    plt.close()

    return mean_mv, std_mv, max_mv
