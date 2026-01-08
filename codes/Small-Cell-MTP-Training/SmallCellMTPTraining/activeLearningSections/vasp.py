import os
import numpy as np

# import regex as re  # Removed: regex no longer needed
import subprocess
import math
import time


from SmallCellMTPTraining.templates import templates as templates
from SmallCellMTPTraining.templates import properties as properties
from SmallCellMTPTraining.io import writers as wr
from SmallCellMTPTraining.io import parsers as pa

# YM note: see qe.py for the quantum expresso implementation. the vasp implementation should be similar
# Note the use of a "template" file 


def generateInitialDataset(inputFolder: str, outputFolder: str, config: dict):
    # Extract the base data sets from the configurations
    baseStrains = np.arange(
        config["baseStrains"][0],
        config["baseStrains"][1],
        config["baseStrainStep"],
    )

    maxCPUs = config["maxProcs"]
    coresPerVASP = 1
    cpusUsed = 0
    os.environ["OMP_NUM_THREADS"] = "1"
    subprocesses = []
    exitCodes = []
    completed = set()

    # Generate and submit the hydrostatic strain runs
    for strain in baseStrains:

        # This is essentially a primitive scheduler/semaphore
        if cpusUsed + coresPerQE > maxCPUs:
            available = False
            while not available:
                time.sleep(1)
                for i, p in enumerate(subprocesses):
                    if i in completed:
                        continue
                    exitCode = p.poll()
                    if not exitCode == None:
                        available = True
                        cpusUsed -= coresPerQE
                        completed.add(i)

        workingFolder = os.path.join(inputFolder, "baseStrain" + str(round(strain, 3)))
        inputFile = os.path.join(
            workingFolder, "baseStrain" + str(round(strain, 3)) + ".in"
        )
        outputFile = os.path.join(
            outputFolder, "baseStrain" + str(round(strain, 3)) + ".out"
        )

        os.mkdir(workingFolder)

        # Prepare VASP input properties
        vaspProperties = {
            "atomPositions": np.array(
                [
                    [0, 0, 0],
                    [
                        0.5 * config["baseLatticeParameter"] * strain,
                        0.5 * config["baseLatticeParameter"] * strain,
                        0.5 * config["baseLatticeParameter"] * strain,
                    ],
                ]
            ),
            "atomTypes": [0, 0],
            "superCell": np.array(
                [
                    [strain * config["baseLatticeParameter"], 0, 0],
                    [0, strain * config["baseLatticeParameter"], 0],
                    [0, 0, strain * config["baseLatticeParameter"]],
                ]
            ),
            "kPoints": [10, 10, 10],
            "ecutwfc": 90,
            "ecutrho": 450,
            "qeOutDir": workingFolder,
            "elements": config["elements"],
            "atomicWeights": config["atomicWeights"],
            "pseudopotentials": config["pseudopotentials"],
            "pseudopotentialDirectory": config["pseudopotentialDirectory"],
        }

        # Write the input and run
        wr.writeVASPInput(inputFile, vaspProperties)
        cpusUsed += coresPerVASP
        subprocesses.append(
            subprocess.Popen(
                "mpirun -np "
                + str(coresPerVASP)
                + inputFile
                + " > "
                + outputFile,
                shell=True,
                cwd=workingFolder,
            ),
        )

    exitCodes = [p.wait() for p in subprocesses]
    return exitCodes
