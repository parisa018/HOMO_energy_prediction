import ase
from ase.io import read
from dscribe.descriptors import MBTR
from ase import Atoms
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rank',type=int,help="rank of the job")
parser.add_argument('--total',type=int,help="total number of jobs")
args = parser.parse_args()

# read the coordinates and element names
mols = read('../QM9_data.xyz',index=':')

# read the homo energy values
with open('../HOMO_QM9.txt', 'r') as f:
    # Read the entire contents of the file into a string
    file_contents = f.read()
# Split the string into a list of lines
homo_energies = file_contents.split('\n')

atoms_names=[]
# get all the atom symbols
for mol in mols:
    atoms_pos=[]
    for i in mol:
        atoms_names.append(i.symbol)
   # Setup the MBTR
# n_k1=20
# sigma_k1=0.1

n_k2=20
sigma_k2=0.1

# n_k3=20
# sigma_k3=0.1
mbtr = MBTR(
    species=set(atoms_names),
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": -1,"max": 3, "n": n_k2, "sigma": sigma_k2},
        "weighting": {"function": "unity", "scale": 0.5, "threshold": 1e-3},
    },
    flatten=True,
    sparse=False

)
filename = f"moleculeDic_{args.rank}.pkl"

with open(filename, 'wb') as f:

    for j,mol in enumerate(mols):
        if(j%args.total==args.rank):
            # each molecule has its own dictionary
            molecule_dic={}

            # get the position of all atoms in a molecule and save them with the key "coordination"
            atoms_pos=[i.position for i in mol]
            molecule_dic['coordination']=atoms_pos

            # get the name of each molecule and save it in the dictionary with the key "name"
            molecule_dic['name']=[i for i in mol.symbols]

            # create the mbtr of the molecule and save it with the key "mbtr"
            my_molecule=Atoms(mol.symbols,atoms_pos)
            mol_mbtr=mbtr.create(my_molecule)
            molecule_dic['mbtr']=mol_mbtr
            
            # add the homo energy to the dictionary
            molecule_dic['homo_energy']=homo_energies[j]

            # dump the pickle file     
            pickle.dump(molecule_dic,f)
