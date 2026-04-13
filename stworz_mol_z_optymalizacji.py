import streamlit as st
import plotly.graph_objects as go
import webbrowser
import py3Dmol
import pandas as pd
import os
import glob
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDetermineBonds, Draw
import subprocess
from rdkit.Chem import rdmolfiles  
import shutil
import re
from natsort import natsorted
from PIL import Image
import py3Dmol
import streamlit.components.v1 as components
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem
import numpy as np 




def stworz_mol_z_XYZ(mol_start, xyz_text):
    if mol_start is None or not xyz_text:
        return None
    try:
        # 1. Tworzymy nowy obiekt na bazie startowego (żeby mieć te same wiązania)
        new_mol = Chem.Mol(mol_start)
        
        # 2. Tworzymy obiekt pomocniczy z samego XYZ, aby wyciągnąć współrzędne
        num_atoms = len(xyz_text.strip().split('\n'))
        xyz_mol = Chem.MolFromXYZBlock(f"{num_atoms}\n\n{xyz_text}")
        
        # 3. Pobieramy konformację z XYZ
        conf = xyz_mol.GetConformer()
        
        # 4. Podmieniamy współrzędne w naszym nowym obiekcie
        # WAŻNE: Zakładamy, że kolejność atomów w XYZ z Gaussiana jest taka sama jak w mol_start
        new_conf = new_mol.GetConformer()
        for i in range(num_atoms):
            new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
        
        new_mol.AddConformer(new_conf, assignId=True)
        return new_mol
    except Exception as e:
        print(f"Błąd konwersji: {e}")
        return None
