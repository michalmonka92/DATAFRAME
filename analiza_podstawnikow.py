# -*- coding: utf-8 -*-
"""
Moduł analizy podstawników - analiza_podstawnikow.py
"""
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolDescriptors, rdDetermineBonds, Draw, rdDepictor
import pandas as pd
import re

def wykonaj_analize_L2(df, path_xyz):
    # 1. Przygotowanie rdzenia
    raw_core = Chem.MolFromXYZFile(path_xyz)
    core_mol = Chem.Mol(raw_core)
    rdDetermineBonds.DetermineConnectivity(core_mol)
    core_no_h = Chem.RemoveHs(core_mol)
    core_smiles = Chem.MolToSmiles(core_no_h)
    
    # 2. Filtrowanie serii L2
    df_l2 = df[df['ID'].str.contains('L2', na=False)].copy()
    if df_l2.empty:
        return None, None, None

    # 3. Pętla analizy (Twoja logika)
    final_results = []
    mols_to_draw = []
    legends = []

    for idx, row in df_l2.iterrows():
        mol_raw = row['S0_MOL_Opt']
        mol_id = row['ID']
        
        if mol_raw is None:
            continue

        m = Chem.RemoveHs(mol_raw)
        c = Chem.RemoveHs(Chem.MolFromSmiles(core_smiles))

        # MCS i odejmowanie
        res = rdFMCS.FindMCS([m, c], completeRingsOnly=True, ringMatchesRingOnly=True)
        if res.smartsString:
            common_mol = Chem.MolFromSmarts(res.smartsString)
            diff = Chem.ReplaceCore(m, common_mol)
            
            wzor = "H"
            smi_clean = "H"
            
            if diff:
                smi_raw = Chem.MolToSmiles(diff).replace('[*]', '').replace('*', '')
                smi_clean = re.sub(r'\[\d+\]', '', smi_raw)
                
                if smi_clean:
                    sub_mol = Chem.MolFromSmiles(smi_clean)
                    if sub_mol:
                        # Wzór sumaryczny
                        wzor = rdMolDescriptors.CalcMolFormula(Chem.AddHs(sub_mol))
                        # Przygotowanie do rysowania
                        rdDepictor.Compute2DCoords(sub_mol)
                        mols_to_draw.append(sub_mol)
                        legends.append(f"{mol_id} ({wzor})")
            
            final_results.append({
                "ID": mol_id, 
                "Podstawnik_Wzor": wzor, 
                "Subst_SMILES": smi_clean
            })

    df_res = pd.DataFrame(final_results)
    
    # Generowanie siatki (grid)
    grid_img = None
    if mols_to_draw:
        grid_img = Draw.MolsToGridImage(
            mols_to_draw, molsPerRow=4, subImgSize=(250, 250), 
            legends=legends, useSVG=False
        )

    return df_res, grid_img, legends