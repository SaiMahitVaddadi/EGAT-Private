
def write_slurm_script(name,output_filename,folder='/scratch/gilbreth/svaddadi/Molecule-Benchmarks/EGAT_Runs/PCQM/',gpus=1,cpus=16,mem=400,partition='standby',time="4:00:00",modules = ["anaconda/2024.02-py311", "gcc/12.3.0", "cuda/11.7.0"],EGAT_dir='/depot/bsavoie/data/Mahit-TS-Energy-Project/GitHub/EGAT'):
    slurm_script = f"""#!/bin/bash
    #
    #SBATCH --job-name={name}
    #SBATCH --output={folder}out/{name}.out
    #SBATCH --error={folder}err/{name}.out
    #SBATCH -A {partition}
    #SBATCH --nodes=1
    """
    if gpus > 0:
        slurm_script += f"""#SBATCH --gpus-per-node={gpus}
    #SBATCH --cpus-per-gpu={cpus}
    """
    else:
        slurm_script += f"""#SBATCH --cpus-per-task={cpus}
    """

    slurm_script += f"""#SBATCH --ntasks-per-node=10
    #SBATCH --mem={str(mem)}G
    #SBATCH -t {time}

    echo Running on host `hostname`
    echo Time is `date`
    
    conda activate EGAT
    """
    for module in modules:
        slurm_script += f"module load {module}\n"

    slurm_script += f"""
    dir={EGAT_dir}
    python $dir/train.py --config {output_filename}
    """
    script_filename = f"{folder}submit/{name}.submit"
    with open(script_filename, 'w') as file:
        file.write(slurm_script)
    print(f"Slurm script saved to {script_filename}")


