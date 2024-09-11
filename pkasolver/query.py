# imports
import logging
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter
from os import path
from typing import Optional

import cairosvg
import numpy as np
import svgutils.transform as sg
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFMCS
from torch_geometric.loader import DataLoader
from IPython.display import SVG

from pkasolver import run_with_mol_list as call_dimorphite_dl
from pkasolver.chem import create_conjugate
from pkasolver.constants import EDGE_FEATURES, NODE_FEATURES
from pkasolver.data import (
    calculate_nr_of_features,
    make_features_dicts,
    mol_to_paired_mol_data,
)
from pkasolver.ml import dataset_to_dataloader, get_device
from pkasolver.ml_architecture import GINPairV1


@dataclass
class Protonation:
    pka: float
    pka_stddev: float
    protonated_mol: Chem.Mol
    deprotonated_mol: Chem.Mol
    reaction_center_idx: int
    ph7_mol: Chem.Mol


logger = logging.getLogger(__name__)

RDLogger.DisableLog("rdApp.*")

node_feat_list = [
    "element",
    "formal_charge",
    "hybridization",
    "total_num_Hs",
    "aromatic_tag",
    "total_valence",
    "total_degree",
    "is_in_ring",
    "reaction_center",
    "smarts",
]
edge_feat_list = ["bond_type", "is_conjugated", "rotatable"]

num_node_features = calculate_nr_of_features(node_feat_list)
num_edge_features = calculate_nr_of_features(edge_feat_list)

# make dicts from selection list to be used in the processing step
selected_node_features = make_features_dicts(NODE_FEATURES, node_feat_list)
selected_edge_features = make_features_dicts(EDGE_FEATURES, edge_feat_list)


class QueryModel:
    def __init__(self, device_str: str = "cuda") -> None:
        self.models = []
        self.device = get_device(device_str)

        for i in range(25):
            model = GINPairV1(
                num_node_features,
                num_edge_features,
                hidden_channels=96,
                device_str=device_str,
            )
            base_path = path.dirname(__file__)
            checkpoint = torch.load(
                f"{base_path}/trained_model_without_epik/best_model_{i}.pt",
                map_location=self.device,
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(device=self.device)
            self.models.append(model)

    def predict_pka_value(self, loader: DataLoader) -> np.ndarray:
        """
        ----------
        loader
            data to be predicted
        Returns
        -------
        np.array
            list of predicted pKa values
        """

        results = []
        assert len(loader) == 1
        for data in loader:  # Iterate in batches over the training dataset.
            data.to(device=self.device)
            consensus_r = []
            for model in self.models:
                y_pred = (
                    model(
                        x_p=data.x_p,
                        x_d=data.x_d,
                        edge_attr_p=data.edge_attr_p,
                        edge_attr_d=data.edge_attr_d,
                        data=data,
                    )
                    .reshape(-1)
                    .detach()
                )

                consensus_r.append(y_pred.tolist())
            results.extend(
                (
                    float(np.average(consensus_r, axis=0)),
                    float(np.std(consensus_r, axis=0)),
                )
            )
        return results


def _get_reaction_centers(
    rdmols: list[Chem.Mol], reference_rdmol: Chem.Mol
) -> list[int]:
    """Takes a list of mol objects of different protonation states,
    and returns the protonation center index

    """
    reaction_centers = []
    for rdmol in rdmols:
        assert reference_rdmol.GetNumAtoms() == rdmol.GetNumAtoms()

        # find MCS
        mcs = rdFMCS.FindMCS(
            [reference_rdmol, rdmol],
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            timeout=120,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )

        # convert from SMARTS
        mcsp = Chem.MolFromSmarts(mcs.smartsString, False)
        ref_rdmol_idxs = reference_rdmol.GetSubstructMatch(mcsp)
        rdmol_idxs = rdmol.GetSubstructMatch(mcsp)

        for i, j in zip(ref_rdmol_idxs, rdmol_idxs):
            if i != j:  # matching not sucessfull
                break
            if (
                reference_rdmol.GetAtomWithIdx(i).GetFormalCharge()
                != rdmol.GetAtomWithIdx(j).GetFormalCharge()
            ):
                reaction_centers.append(i)

    logger.debug(set(reaction_centers))
    return reaction_centers


def _sort_conj(rdmols: list[Chem.Mol]) -> list[Chem.Mol]:
    """Sort two molecules by the number of hydrogens,
    so that the protonated molecule is first in the list."""
    assert len(rdmols) == 2

    num_hydrogens = [
        sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()]) for mol in rdmols
    ]

    if num_hydrogens[0] - num_hydrogens[1] == 1:
        return rdmols
    elif num_hydrogens[0] - num_hydrogens[1] == -1:
        return rdmols[::-1]
    else:
        raise RuntimeError(
            "Neighboring protonation states are only allowed to have a difference of a"
            " single hydrogen."
        )


def _deduplicate_protonations(protonations: list[Protonation]) -> list[Protonation]:
    """check whether two states have the same pKa value and remove one of them"""
    unique_protonations = {}
    logger.debug(protonations)
    for protonation in protonations:
        m1, m2 = _sort_conj([protonation.protonated_mol, protonation.deprotonated_mol])
        unique_protonations[hash((Chem.MolToSmiles(m1), Chem.MolToSmiles(m2)))] = (
            protonation
        )
    # logger.debug([all_r[k] for k in sorted(all_r, key=all_r.get)])
    return sorted(list(unique_protonations.values()), key=attrgetter("pka"))


def _get_protonation(
    protonated_rdmol: Chem.Mol,
    deprotonated_rdmol: Chem.Mol,
    reaction_center_idx: int,
    query_model: QueryModel,
    rdmol_at_ph_7: Chem.Mol,
) -> Protonation:
    m = mol_to_paired_mol_data(
        protonated_rdmol,
        deprotonated_rdmol,
        reaction_center_idx,
        selected_node_features,
        selected_edge_features,
    )
    loader = dataset_to_dataloader([m], 1)
    pka, pka_std = query_model.predict_pka_value(loader)

    return Protonation(
        pka,
        pka_std,
        protonated_rdmol,
        deprotonated_rdmol,
        reaction_center_idx,
        ph7_mol=rdmol_at_ph_7,
    )


def _get_protonations_unidirection(
    rdmol: Chem.Mol,
    reaction_center_idxs: list[int],
    target_ph: float,
    query_model: QueryModel,
    rdmol_at_ph_7: Chem.Mol,
) -> list[Protonation]:
    conjugate_states = []
    rdmol_at_state = deepcopy(rdmol)
    used_reaction_center_idxs = deepcopy(reaction_center_idxs)

    logger.debug(f"Creating conjugates for target pH {target_ph} ...")

    for _ in reaction_center_idxs:
        states_per_iteration = []
        for i in used_reaction_center_idxs:
            try:
                conj = create_conjugate(
                    rdmol_at_state,
                    i,
                    pka=target_ph,
                    known_pka_values=False,
                )
            except Exception:
                continue

            protonated_rdmol, deprotonated_rdmol = _sort_conj([conj, rdmol_at_state])

            pka_state = _get_protonation(
                protonated_rdmol,
                deprotonated_rdmol,
                i,
                query_model,
                rdmol_at_ph_7,
            )

            if pka_state.pka < 0.5 or pka_state.pka > 13.5:
                logger.debug("pKa value out of bound!")
                continue

            # If the new state is not closer to the target pH, skip
            if conjugate_states and not (
                (target_ph < 7.0 and pka_state.pka < conjugate_states[-1].pka)
                or (target_ph >= 7.0 and pka_state.pka > conjugate_states[-1].pka)
            ):
                continue

            states_per_iteration.append(pka_state)

        if not states_per_iteration:
            break

        # get the protonation state with the most neutral pka
        if target_ph < 7.0:
            neutral_pka_state = max(states_per_iteration, key=attrgetter("pka"))
            rdmol_at_state = deepcopy(neutral_pka_state.protonated_mol)
        else:
            neutral_pka_state = min(states_per_iteration, key=attrgetter("pka"))
            rdmol_at_state = deepcopy(neutral_pka_state.deprotonated_mol)
        conjugate_states.append(neutral_pka_state)
        # avoid double protonation
        used_reaction_center_idxs.remove(conjugate_states[-1].reaction_center_idx)

    return conjugate_states


def calculate_microstate_pka_values(
    mol: Chem.rdchem.Mol,
    only_dimorphite: bool = False,
    query_model: Optional[QueryModel] = None,
    verbose: bool = False,
    device_str: str = "cuda",
) -> list[Protonation]:
    """Enumerate protonation states using a rdkit mol as input"""

    if query_model is None:
        query_model = QueryModel(device_str=device_str)

    rdmols_at_ph_7 = call_dimorphite_dl([mol], min_ph=7.0, max_ph=7.0, pka_precision=0)
    assert len(rdmols_at_ph_7) == 1
    rdmol_at_ph_7 = rdmols_at_ph_7[0]

    if verbose:
        logger.info(f"Proposed mol at pH 7.4: {Chem.MolToSmiles(rdmol_at_ph_7)}")
        logger.info("Using dimorphite-dl to identify protonation sites.")
    all_mols = call_dimorphite_dl([mol], min_ph=0.5, max_ph=13.5)

    if only_dimorphite:
        logger.warning(
            "BEWARE! This is experimental and might generate wrong protonation states."
        )
        # sort mols
        # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
        num_hydrogens = [
            sum(atom.GetTotalNumHs() for atom in mol.GetAtoms()) for mol in all_mols
        ]
        mols_sorted = [x for _, x in sorted(zip(num_hydrogens, all_mols), reverse=True)]

        reaction_center_idxs = _get_reaction_centers(mols_sorted, mols_sorted[0])
        # return only mol pairs
        protonations = []
        for nr_of_states, idx in enumerate(reaction_center_idxs):
            protonated_rdmol = mols_sorted[nr_of_states]
            deprotonated_rdmol = mols_sorted[nr_of_states + 1]

            logger.debug(Chem.MolToSmiles(protonated_rdmol))
            logger.debug(Chem.MolToSmiles(deprotonated_rdmol))

            protonation = _get_protonation(
                protonated_rdmol,
                deprotonated_rdmol,
                idx,
                query_model,
                rdmol_at_ph_7,
            )

            protonations.append(protonation)
        logger.debug(protonations)
    else:
        reaction_center_idxs = sorted(
            list(set(_get_reaction_centers(all_mols, rdmol_at_ph_7)))
        )

        acidic_protonations = _get_protonations_unidirection(
            rdmol_at_ph_7,
            reaction_center_idxs,
            target_ph=0.0,
            query_model=query_model,
            rdmol_at_ph_7=rdmol_at_ph_7,
        )
        logger.debug(acidic_protonations)

        basic_protonations = _get_protonations_unidirection(
            rdmol_at_ph_7,
            reaction_center_idxs,
            target_ph=14.0,
            query_model=query_model,
            rdmol_at_ph_7=rdmol_at_ph_7,
        )
        logger.debug(basic_protonations)

        protonations = _deduplicate_protonations(
            basic_protonations + acidic_protonations[::-1]
        )

    if not protonations:
        logger.error("Could not identify any ionizable group. Aborting.")

    return protonations


def draw_pka_map(protonations: list[Protonation], size: tuple[int, int] = (450, 450)):
    """draw mol at pH=7.0 and indicate protonation sites with respective pKa values"""
    rdmol_at_ph_7 = deepcopy(protonations[0].ph7_mol)
    for protonation in protonations:
        reaction_center_atom = rdmol_at_ph_7.GetAtomWithIdx(
            protonation.reaction_center_idx
        )
        try:
            reaction_center_atom.SetProp(
                "atomNote",
                f'{reaction_center_atom.GetProp("atomNote")},   {protonation.pka:.2f}',
            )
        except Exception:
            reaction_center_atom.SetProp("atomNote", f"{protonation.pka:.2f}")
    return Draw.MolToImage(rdmol_at_ph_7, size=size)


def draw_pka_reactions(protonations: list, height=250, write_png_to_file: str = ""):
    """
    Draws protonation states.
    file can be saved as png using `write_png_to_file` parameter.
    """
    draw_pairs, pair_atoms, legend = [], [], []
    for i, protonation in enumerate(protonations):
        draw_pairs.extend([protonation.protonated_mol, protonation.deprotonated_mol])
        pair_atoms.extend(
            [[protonation.reaction_center_idx], [protonation.reaction_center_idx]]
        )
        legend.append(
            f"pka_{i} = {protonation.pka:.2f} (stddev: {protonation.pka_stddev:.2f})"
        )

    s = Draw.MolsToGridImage(
        draw_pairs,
        molsPerRow=2,
        subImgSize=(height * 2, height),
        highlightAtomLists=pair_atoms,
        useSVG=True,
    )
    # Draw.MolsToGridImage returns different output depending on whether it is called in a notebook or a script
    if hasattr(s, "data"):
        s = s.data.replace("svg:", "")
    fig = sg.fromstring(s)

    for i, text in enumerate(legend):
        label = sg.TextElement(
            height * 2,
            (height * (i + 1)) - 10,
            text,
            size=14,
            font="sans-serif",
            anchor="middle",
        )
        fig.append(label)

        h = height * (i + 0.5)
        w = height * 2
        for fx_1, fx_2, fy_1, fy_2 in (
            (0.9, 1.1, -0.02, -0.02),
            (1.1, 1.07, -0.02, -0.04),
            (0.9, 1.1, 0.02, 0.02),
            (0.9, 0.93, 0.02, 0.04),
        ):
            fig.append(
                sg.LineElement(
                    [(w * fx_1, h + height * fy_1), (w * fx_2, h + height * fy_2)],
                    width=2,
                    color="black",
                )
            )
    # if png file path is passed write png file
    if write_png_to_file:
        cairosvg.svg2png(
            bytestring=fig.to_str(), write_to=f"{write_png_to_file}", dpi=300
        )
    return SVG(fig.to_str())
