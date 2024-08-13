# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:36:16 2024

Advanced GPU-accelerated Surface Analysis and Visualization of Proteins
Using PyCuda, BioPython, and PyMol

@author: Nasser GHAZI
"""

#%% Import libraries and define paths

import os
import numpy as np
from Bio.PDB import PDBParser

os.environ['PATH'] += r';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
os.environ['PATH'] += r';C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.31.31103\bin\Hostx64\x64'

# Define the file paths
pdb_file = "C:/Users/Nasser/OneDrive/Desktop/Programming/CUDA/GPU capstone/2c0k.pdb"
output_dir = "C:/Users/Nasser/OneDrive/Desktop/Programming/CUDA/GPU capstone"

#%% Parse the PDB file and extract atomic coordinates and radii

# Parse the PDB file
parser = PDBParser()
structure = parser.get_structure('protein', pdb_file)

# Extract atomic coordinates and radii
atoms = []
radii = []
atom_ids = []  # To track the ID for each atom
atom_residues = []  # To track the residue each atom belongs to

# A dictionary to get van der Waals radii based on atom type
vdw_radii = {
    'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8
}

for atom in structure.get_atoms():
    atoms.append(atom.coord)
    atom_type = atom.element
    radii.append(vdw_radii.get(atom_type, 1.5))  # Default radius if not found
    atom_ids.append(atom.get_id())
    atom_residues.append(atom.get_parent().get_id())

# Convert lists to NumPy arrays
atom_coords = np.array(atoms, dtype=np.float32)
atom_radii = np.array(radii, dtype=np.float32)

print(f"Parsed {len(atom_coords)} atoms from the PDB file.")

#%% Visualize and save the initial structure in PyMOL

import pymol
from pymol import cmd

# Start PyMOL
pymol.finish_launching()

# Load the PDB file
cmd.load(pdb_file, "protein_initial")

# Set the initial visualization settings
cmd.show("surface", "protein_initial")
cmd.spectrum("count", "rainbow", "protein_initial")

# Save the initial visualization
cmd.save(os.path.join(output_dir, "protein_initial.pse"))
cmd.png(os.path.join(output_dir, "protein_initial.png"), dpi=300)

print("Initial visualization complete. PyMOL session and image saved.")

#%% Setup and run the GPU-accelerated surface area computation

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Define the CUDA kernel for calculating surface areas
kernel_code = """
__global__ void compute_surface_area(float *atom_coords, float *radii, float *areas, int num_atoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms) return;

    float x_i = atom_coords[3 * i];
    float y_i = atom_coords[3 * i + 1];
    float z_i = atom_coords[3 * i + 2];
    float r_i = radii[i];
    
    float surface_area = 4.0f * 3.14159265359f * r_i * r_i;

    for (int j = 0; j < num_atoms; j++) {
        if (i != j) {
            float x_j = atom_coords[3 * j];
            float y_j = atom_coords[3 * j + 1];
            float z_j = atom_coords[3 * j + 2];
            float r_j = radii[j];

            float dx = x_j - x_i;
            float dy = y_j - y_i;
            float dz = z_j - z_i;
            float dist = sqrt(dx * dx + dy * dy + dz * dz);
            
            if (dist < (r_i + r_j)) {
                float overlap_area = 2.0f * 3.14159265359f * r_j * (r_j - (dist - r_i + r_j) / 2.0f);
                surface_area -= overlap_area;
            }
        }
    }
    areas[i] = max(surface_area, 0.0f); // Ensure no negative surface area
}

"""

# Compile the kernel
mod = SourceModule(kernel_code)
compute_surface_area = mod.get_function("compute_surface_area")

# Allocate GPU memory and transfer data
num_atoms = atom_coords.shape[0]
atom_coords_gpu = cuda.mem_alloc(atom_coords.nbytes)
radii_gpu = cuda.mem_alloc(atom_radii.nbytes)
areas_gpu = cuda.mem_alloc(num_atoms * np.float32().nbytes)

cuda.memcpy_htod(atom_coords_gpu, atom_coords)
cuda.memcpy_htod(radii_gpu, atom_radii)

# Define grid and block dimensions
block_size = 128
grid_size = (num_atoms + block_size - 1) // block_size

# Call the kernel
compute_surface_area(
    atom_coords_gpu, radii_gpu, areas_gpu,
    np.int32(num_atoms),
    block=(block_size, 1, 1), grid=(grid_size, 1, 1)
)

# Retrieve the results from the GPU
areas = np.empty(num_atoms, dtype=np.float32)
cuda.memcpy_dtoh(areas, areas_gpu)

# Save the results
output_path = os.path.join(output_dir, "surface_areas.txt")
np.savetxt(output_path, areas, header="Surface Areas per Atom")

print(f"Computed surface areas saved to {output_path}.")

#%% Quantitative Analysis

# Identify atoms with extreme surface areas
max_area = np.max(areas)
min_area = np.min(areas)
max_area_atom = np.argmax(areas)
min_area_atom = np.argmin(areas)

print(f"Maximum surface area: {max_area:.2f}, Atom ID: {atom_ids[max_area_atom]}, Residue: {atom_residues[max_area_atom]}")
print(f"Minimum surface area: {min_area:.2f}, Atom ID: {atom_ids[min_area_atom]}, Residue: {atom_residues[min_area_atom]}")

# Optional: Sum surface areas by residue
residue_surface_areas = {}
for i, res in enumerate(atom_residues):
    if res in residue_surface_areas:
        residue_surface_areas[res] += areas[i]
    else:
        residue_surface_areas[res] = areas[i]

# Identify residue with maximum surface area
max_residue = max(residue_surface_areas, key=residue_surface_areas.get)
print(f"Residue with maximum surface area: {max_residue}, Surface area: {residue_surface_areas[max_residue]:.2f}")

#%% Visualize the results in PyMOL (after computation)

# Reload the PDB file into a new object for the "after" visualization
cmd.load(pdb_file, "protein_processed")

# Color atoms based on their computed surface areas
cmd.alter("all", "b = 0.0")  # Reset B-factor
for i, area in enumerate(areas):
    cmd.alter(f"id {i+1}", f"b = {area}")

# Set visualization settings
cmd.show("surface", "protein_processed")
cmd.spectrum("b", "rainbow", "protein_processed", minimum=areas.min(), maximum=areas.max())

# Highlight the atom with the maximum surface area
cmd.select("max_area_atom", f"id {max_area_atom+1}")
cmd.show("spheres", "max_area_atom")
cmd.color("red", "max_area_atom")

# Save the processed visualization
cmd.save(os.path.join(output_dir, "protein_surface.pse"))
cmd.png(os.path.join(output_dir, "protein_surface.png"), dpi=300)

print("Processed visualization complete. PyMOL session and image saved.")


#%% Plot Histogram of Surface Areas

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))
plt.hist(areas, bins=30, color='blue', edgecolor='black')
plt.title('Histogram of Atom Surface Areas')
plt.xlabel('Surface Area (Å²)')
plt.ylabel('Count')
plt.grid(True)

# Save the histogram
histogram_path = os.path.join(output_dir, "surface_area_histogram.png")
plt.savefig(histogram_path, dpi=300)

print(f"Histogram of surface areas saved to {histogram_path}.")
