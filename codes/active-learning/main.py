import os 
import numpy as np
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt 
import datetime
import shutil



import SmallCellMTPTraining.io.writers 

from plot_extrapolation_grade_hist import plot_extrapolation_grade_hist
from cfg_parser import extract_cfgs_from_file, generate_vasp_inputs, run_vasp_files, plot_train_history, get_sorted_folders
def generate_lamnps_file(pot_file, outdir):

    md_properties_single = {
        "latticeParameter": 2.831, # angstroms 
        "boxDimensions": [1, 1, 1],
        "potFile": pot_file,
        "temperature": 300, # K
        "elements": ["Fe"],
        "atomicWeights": [55.845],
        "pressure":1, # [bar] 
        "threshold_break":10,
        "threshold_save":0.1,
        "preselected_cfg":f"{outdir}/preselected.cfg"
    }
    SmallCellMTPTraining.io.writers.writeMDInput_fcc(f"{outdir}/lammps.in", md_properties_single)

def lammps_phase(iter:int):
    # initial trained 
    if iter == 0:
        almtp_file = "/global/home/hpc5626/mtp/Fe/L18/mtp"
    else:
        # load from the previous iteration!
        almtp_file = f"/global/home/hpc5626/mtp/Fe/L18/oldPot/{iter}"

    os.makedirs(f'iter/{iter}', exist_ok=True)

    # # generate input file 
    print(f'Generating LAMMPS input file at: iter/{iter}', flush=True)
    generate_lamnps_file(almtp_file, f'iter/{iter}')

    # # # run lammps 
    print(f'Running LAMMPS', flush=True)
    run_log = subprocess.run([f"mpirun -np 4 /global/home/hpc5626/MTP/interface-lammps-mlip-3/lmp_mpi -in iter/{iter}/lammps.in"], shell=True, capture_output=True)
    shutil.move('preselected.cfg', f'iter/{iter}/preselected.cfg')

    with open(f'iter/{iter}/lammps.log','w') as f:
        f.writelines(run_log.stdout.decode())

    # # plot histogram 
    mean_mv, std_mv, max_mv = plot_extrapolation_grade_hist(log_file=f'iter/{iter}/lammps.log',savedir=f'iter/{iter}')
    with open(f'iter/{iter}/mv_hist.txt','w') as f:
        f.write(f'mean: {mean_mv}\n')
        f.write(f'std: {std_mv}\n')
        f.write(f'max: {max_mv}\n')


    # Sample query error 
    sample_query_error(iter, almtp_file, n_sample=200)

    # process 
    process_sample_query_error(iter)

    # Run new threshold configurations 
    run_new_batch_cfg_with_threshold(iter, almtp_file, threshold_save=2.1)

    # Convert the latest OUTCAR files back into mlp configs and begin training 

    train_new_MLIP_model(iter)

    pass


def sample_query_error(iter:int, almtp_file:str, n_sample=200):
    '''
    Proportionately sample queries and see the error 
    '''

    print(f'Generating VASP input files at: iter/{iter}/vasp', flush=True)

    assert os.path.exists(f'iter/{iter}/preselected.cfg')
    cfg_list = extract_cfgs_from_file(f'iter/{iter}/preselected.cfg')

    n_config = len(cfg_list)
    rng = np.random.default_rng()
    cfg_indices = rng.permutation(n_config)[0:n_sample]

    os.makedirs(f'iter/{iter}/vasp', exist_ok=True)

    with open(f'iter/{iter}/sampled_preselected.cfg','w') as f:
        for cfg_idx in cfg_indices:
            cfg = cfg_list[cfg_idx]
            f.writelines(cfg.lines)
            f.write('\n')

    # write this config to OUTCAR file 
    generate_vasp_inputs(selected_cfgs_file=f'iter/{iter}/sampled_preselected.cfg',
                         output_dir=f'iter/{iter}/vasp',
                         incar_file='Fe_vasp/INCAR',
                         potcar_file='Fe_vasp/POTCAR',
                         )
    
    # if everything goes correctly, run VASP! 
    run_vasp_files(folder=f'iter/{iter}/vasp',
                   almtp_file=almtp_file)
    
    print(f'All runs completed', flush=True)

    pass

def process_sample_query_error(iter:int):
    '''To be run after sample_query_error has generated correct files

    Read error metrics, plot some histograms 

    '''

    # get the mv grade
    with open(f'iter/{iter}/sampled_preselected.cfg','r') as f:
        lines = f.readlines()
    
    cfg_mv = []
    for line in lines: 
        if 'MV_grade' in line: 
            tmp = line.split()[-1]
            cfg_mv.append(float(tmp))


    assert os.path.exists(f'iter/{iter}/vasp')

    vasp_files = os.listdir(f'iter/{iter}/vasp')

    energy_diff = []
    forces_rms = []

    # Check that runs have been completed and that a test_eq_cfg.log.0 file exists 
    for vasp in vasp_files: 
        assert os.path.exists(f'iter/{iter}/vasp/{vasp}/test_eq_cfg.log.0')
        with open(f'iter/{iter}/vasp/{vasp}/test_eq_cfg.log.0','r') as f:
            lines = f.readlines()[0]
        
        lines = lines.split()
        energy_diff.append(float(lines[5].split(':')[-1]))
        forces_rms.append(float(lines[8]))

    mean_ene_err = np.mean(energy_diff)
    mean_forces_err = np.mean(forces_rms)

    with open(f'iter/{iter}/sampled_mean_error.txt','w') as f:
        f.write(f'Mean energy error: {mean_ene_err}\n')
        f.write(f'Mean forces error: {mean_forces_err}\n')

    plt.scatter(cfg_mv, energy_diff)
    plt.ylabel(r'abs($E - E_{ref}$)')
    plt.xlabel(f'Extrapolation grade')
    plt.tight_layout()
    plt.savefig(f'iter/{iter}/sampled_exgrade_ene.pdf')
    plt.close()

    plt.scatter(cfg_mv, forces_rms)
    plt.ylabel(r'$|| \vec{F} - \vec{F}_{ref} ||_{2}$')
    plt.xlabel(f'Extrapolation grade')
    plt.tight_layout()
    plt.savefig(f'iter/{iter}/sampled_exgrade_rmseforce.pdf')
    plt.close()


    pass

def run_new_batch_cfg_with_threshold(iter:int, almtp_file:str, threshold_save=2.1):
    '''
    Proportionately sample queries and see the error 
    '''

    print(f'Generating VASP input files at: iter/{iter}/vasp_threshold with threshold value: {threshold_save}', flush=True)

    assert os.path.exists(f'iter/{iter}/preselected.cfg')
    cfg_list = extract_cfgs_from_file(f'iter/{iter}/preselected.cfg')

    cfg_threshold = []


    for cfg in cfg_list:
        if cfg.mv_grade >= threshold_save:
            cfg_threshold.append(cfg)

    print(f'# of cfg with extrapolation grade >= {threshold_save} = {len(cfg_threshold)}')


    os.makedirs(f'iter/{iter}/vasp_threshold', exist_ok=True)

    with open(f'iter/{iter}/threshold_preselected.cfg','w') as f:
        for cfg in cfg_threshold:
            f.writelines(cfg.lines)
            f.write('\n')

    # write this config to OUTCAR file 
    generate_vasp_inputs(selected_cfgs_file=f'iter/{iter}/threshold_preselected.cfg',
                         output_dir=f'iter/{iter}/vasp_threshold',
                         incar_file='Fe_vasp/INCAR',
                         potcar_file='Fe_vasp/POTCAR',
                         )
    
    # if everything goes correctly, run VASP! 
    run_vasp_files(folder=f'iter/{iter}/vasp_threshold',
                   almtp_file=almtp_file)
    
    print(f'All runs completed', flush=True)

    pass

def train_new_MLIP_model(iter:int,mpi=24):

    if iter == 0:
        previous_cfg = ['/global/home/hpc5626/mtp/Fe/L18/mtp/train.cfg']
    else:
        previous_cfg = [f"/global/home/hpc5626/mtp/Fe/L18/mtp//{iter}/train.cfg"]

    os.makedirs(f'iter/{iter}/updated_model', exist_ok=True)

    print(f'Writing new train config from vasp_threshold folder to: iter/{iter}/updated_model/train.cfg', flush = True)

    # remove the previous one first 
    if os.path.exists(f'iter/{iter}/updated_model/train.cfg'):
        print(f'Removing previous train.cfg')
        os.remove(f'iter/{iter}/updated_model/train.cfg')
    new_train_cfg = open(f'iter/{iter}/updated_model/train.cfg','w')
    for prev_cfg in previous_cfg:
        with open(prev_cfg,'r') as f:
            lines = f.readlines()
        new_train_cfg.writelines(lines)
        new_train_cfg.write('\n')

    # first check that the previous runs in 'vasp_threshold' has been completed 
    vasp_folders = get_sorted_folders(dir=f'iter/{iter}/vasp_threshold',startswith='vasp')

    print(f'Processing {len(vasp_folders)} number of VASP OUTCARS', flush=True)
    for vasp in vasp_folders:
        assert os.path.exists(f'iter/{iter}/vasp_threshold/{vasp}/OUTCAR')
        output_log = subprocess.run([f'mlp convert --input_format=outcar iter/{iter}/vasp_threshold/{vasp}/OUTCAR tmp_mlip.cfg'], shell=True, capture_output=False)
        with open('tmp_mlip.cfg','r') as f:
            lines = f.readlines()
        os.remove('tmp_mlip.cfg')
        new_train_cfg.writelines(lines)
        new_train_cfg.write('\n')
    print(f'Finished processing VASP OUTCARS', flush=True)
        
    new_train_cfg.close()

    print(f'Starting training of MLIP model. ')
    # sanity check 
    assert os.path.exists(f'iter/{iter}/updated_model/train.cfg')
    # begin training 
    print(f'Starting training: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Num of MPI: {mpi}', flush=True)

    almtp_file = f'/global/home/hpc5626/MTP/mlip-3/MTP_templates/18.almtp'
    train_cfg_file = f'iter/{iter}/updated_model/train.cfg'
    um_dir = f'iter/{iter}/updated_model'
    output_log = subprocess.run([f"mpirun  -np {mpi} mlp train {almtp_file} {train_cfg_file} --save_to={um_dir}/pot.almtp --iteration_limit=100"], capture_output=True, text=True, shell=True)

    with open(f'{um_dir}/train.log','w') as f:
        f.writelines(output_log.stdout)

    print(f'Training completed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


    fig = plot_train_history(f'{um_dir}/train.log')
    plt.savefig(f'{um_dir}/train_hist.pdf')
    plt.close(fig)

    pass





if __name__ == "__main__":

    # train model 23

    # train_new_MLIP_model(iter=23)

    for iter in [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]:
        lammps_phase(iter=iter)
    # lammps_phase(iter=21)
    # lammps_phase(iter=22)
    # lammps_phase(iter=23)
    pass
