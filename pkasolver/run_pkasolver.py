#!/db2/users/chem_pipeline/anaconda3/envs/pkasolver/bin/python

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from operator import attrgetter

import numpy as np
import pandas as pd
import torch
from openeye import oechem, oequacpac
from pkasolver.chem import create_conjugate
from pkasolver.constants import DEVICE, EDGE_FEATURES, NODE_FEATURES
from pkasolver.data import (
    calculate_nr_of_features,
    make_features_dicts,
    mol_to_paired_mol_data,
)
from pkasolver.ml import dataset_to_dataloader
from pkasolver.ml_architecture import GINPairV1
from rdkit import Chem, RDLogger
from torch_geometric.loader import DataLoader


@dataclass
class States:
    pka: float
    pka_stddev: float
    protonated_mol: Chem.Mol
    deprotonated_mol: Chem.Mol
    reaction_center_idx: int
    ph7_mol: Chem.Mol


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
    def __init__(self):
        self.models = []

        for i in range(25):
            # TODO: prepare CPU model
            model_name, model_class = "GINPair", GINPairV1
            model = model_class(
                num_node_features, num_edge_features, hidden_channels=96
            ).to("cpu")
            base_path = os.path.dirname(__file__)
            if torch.cuda.is_available() == False:  # If only CPU is available
                checkpoint = torch.load(
                    f"{base_path}/pkasolver/trained_model_without_epik/best_model_{i}.pt",
                    map_location=torch.device("cpu"),
                )
            else:
                checkpoint = torch.load(
                    f"{base_path}/pkasolver/trained_model_without_epik/best_model_{i}.pt"
                )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model.to(device=DEVICE)
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
            data.to(device=DEVICE)
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


def _get_ionization_indices(mol_list: list, compare_to: Chem.Mol) -> list:
    """Takes a list of mol objects of different protonation states,
    and returns the protonation center index

    """
    from rdkit.Chem import rdFMCS

    list_of_reaction_centers = []
    for idx, m2 in enumerate(mol_list):
        m1 = compare_to
        assert m1.GetNumAtoms() == m2.GetNumAtoms()

        # find MCS
        mcs = rdFMCS.FindMCS(
            [m1, m2],
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            timeout=120,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )

        # convert from SMARTS
        mcsp = Chem.MolFromSmarts(mcs.smartsString, False)
        s1 = m1.GetSubstructMatch(mcsp)
        s2 = m2.GetSubstructMatch(mcsp)

        for i, j in zip(s1, s2):
            # if i != j:  # matching not sucessfull
            #    break
            a1 = m1.GetAtomWithIdx(i)
            a2 = m2.GetAtomWithIdx(j)
            c1 = a1.GetFormalCharge()
            c2 = a2.GetFormalCharge()
            n_h1 = a1.GetNumExplicitHs()
            n_h2 = a2.GetNumExplicitHs()
            if n_h1 != n_h2 and c1 != c2:
                list_of_reaction_centers.append(i)

            if (
                m1.GetAtomWithIdx(i).GetFormalCharge()
                != m2.GetAtomWithIdx(j).GetFormalCharge()
            ):
                list_of_reaction_centers.append(i)

    return list_of_reaction_centers


def _sort_conj(mols: list):
    """sort mols based on number of hydrogen"""

    assert len(mols) == 2
    nr_of_hydrogen = [
        np.sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()]) for mol in mols
    ]
    if abs(nr_of_hydrogen[0] - nr_of_hydrogen[1]) != 1:
        raise RuntimeError(
            "Neighboring protonation states are only allowed to have a difference of a single hydrogen."
        )
    mols_sorted = [
        x for _, x in sorted(zip(nr_of_hydrogen, mols), reverse=True)
    ]  # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    return mols_sorted


def _check_for_duplicates(states: list):
    """check whether two states have the same pKa value and remove one of them"""
    all_r = dict()
    for state in states:
        m1, m2 = _sort_conj([state.protonated_mol, state.deprotonated_mol])
        all_r[hash((Chem.MolToSmiles(m1), Chem.MolToSmiles(m2)))] = state
    return sorted([all_r[k] for k in all_r], key=attrgetter("pka"))


def get_pka_info(states):
    """Get pKa information for each state."""
    pka_info = []
    for state in sorted(states, key=lambda x: abs(x.pka - 7.4)):
        proto_mol = Chem.MolToSmiles(state.protonated_mol)
        deproto_mol = Chem.MolToSmiles(state.deprotonated_mol)
        pka_info.append(
            f"{state.reaction_center_idx};"
            f"{state.pka:.2f};"
            f"{state.pka_stddev:.2f};"
            f"{proto_mol};"
            f"{deproto_mol}"
        )
    return pka_info


def remove_highly_charged_mols(mols):
    # remove highly charged mols
    chgs = []
    for a in mols:
        chg = abs(oechem.OENetCharge(a))
        chgs.append(chg)
    chgs = sorted(set(chgs))
    if len(chgs) > 1:
        if chgs[0] < 2:
            chg_cut = 2
        else:
            chg_cut = chgs[0]
        ret = []
        for a in mols:
            chg = abs(oechem.OENetCharge(a))
            if chg <= chg_cut:
                ret.append(a)
        mols = ret
    return mols


def enumerate_protonation_states(states):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = f"{tmpdir}/input.smi"
        output_file = f"{tmpdir}/output.smi"

        f = oechem.oemolostream(input_file)
        for mols in states:
            for mol in mols:
                oechem.OEWriteMolecule(f, mol)

        cmd = (
            f"cd {tmpdir}; pkatyper -in {input_file} -out {output_file}"
            f" -warts false -max 64 2> /dev/null"
        )
        subprocess.run(cmd, shell=True)

        f = oechem.oemolistream(output_file)
        mols = {}
        for mol in f.GetOEGraphMols():
            idx = int(mol.GetTitle())
            if idx not in mols:
                mols[idx] = []
            mols[idx].append(mol.CreateCopy())

        for idx in mols:
            if len(mols[idx]) <= 1:
                continue
            mols[idx] = remove_highly_charged_mols(mols[idx])

    return mols


def get_protonation_states(
    neutral_states, protonated_states, min_ph, max_ph, query_model=None
):
    if query_model == None:
        query_model = QueryModel()

    for mols_ph7 in neutral_states:
        mol_total = []
        for mol_ph7 in mols_ph7:
            title_idx = int(mol_ph7.GetTitle())
            # change OEMol to RDMol
            mol_ph7 = oemol_to_rdmol(mol_ph7)
            all_mols = [oemol_to_rdmol(m) for m in protonated_states[title_idx]]

            # print('proposed NEUTRAL PROTOMER:', Chem.MolToSmiles(mol_ph7))
            # print('len(all_mols)', len(all_mols))

            # identify protonation sites
            reaction_center_atom_idxs = sorted(
                set(_get_ionization_indices(all_mols, mol_ph7))
            )
            # print('reaction_center_atom_idxs', reaction_center_atom_idxs)
            mols = [mol_ph7]

            acids = []
            mol_at_state = deepcopy(mol_ph7)

            used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
            # print("Start with acids ...")
            # for each possible protonation state
            for _ in reaction_center_atom_idxs:
                states_per_iteration = []
                # for each possible reaction center
                for i in used_reaction_center_atom_idxs:
                    try:
                        conj = create_conjugate(
                            mol_at_state,
                            i,
                            pka=0.0,
                            known_pka_values=False,
                        )
                    except:
                        continue

                    # sort mols (protonated/deprotonated)
                    sorted_mols = _sort_conj([conj, mol_at_state])

                    m = mol_to_paired_mol_data(
                        sorted_mols[0],
                        sorted_mols[1],
                        i,
                        selected_node_features,
                        selected_edge_features,
                    )
                    # calc pka value
                    loader = dataset_to_dataloader([m], 1)
                    pka, pka_std = query_model.predict_pka_value(loader)
                    pair = States(
                        pka,
                        pka_std,
                        sorted_mols[0],
                        sorted_mols[1],
                        reaction_center_idx=i,
                        ph7_mol=mol_ph7,
                    )

                    # test if pka is inside pH range
                    if pka < min_ph:
                        # skip rest
                        continue

                    # print(
                    #    "acid: ",
                    #    pka,
                    #    Chem.MolToSmiles(conj),
                    #    i,
                    #    Chem.MolToSmiles(mol_at_state),
                    # )

                    # if this is NOT the first state found
                    if acids:
                        # check if previous pka value is lower and if yes, add it
                        if pka < acids[-1].pka:
                            states_per_iteration.append(pair)
                    else:
                        # if this is the first state found
                        states_per_iteration.append(pair)

                if not states_per_iteration:
                    # no protonation state left
                    break

                # get the protonation state with the highest pka
                acids.append(max(states_per_iteration, key=attrgetter("pka")))
                used_reaction_center_atom_idxs.remove(
                    acids[-1].reaction_center_idx
                )  # avoid double protonation
                mol_at_state = deepcopy(acids[-1].protonated_mol)

            # print('acids', acids)

            #######################################################
            # continue with bases
            #######################################################

            bases = []
            mol_at_state = deepcopy(mol_ph7)
            # print("Start with bases ...")
            used_reaction_center_atom_idxs = deepcopy(reaction_center_atom_idxs)
            # print(reaction_center_atom_idxs)
            # for each possible protonation state
            for _ in reaction_center_atom_idxs:
                states_per_iteration = []
                # for each possible reaction center
                for i in used_reaction_center_atom_idxs:
                    try:
                        conj = create_conjugate(
                            mol_at_state, i, pka=13.5, known_pka_values=False
                        )
                    except:
                        continue
                    sorted_mols = _sort_conj([conj, mol_at_state])
                    m = mol_to_paired_mol_data(
                        sorted_mols[0],
                        sorted_mols[1],
                        i,
                        selected_node_features,
                        selected_edge_features,
                    )
                    # calc pka values
                    loader = dataset_to_dataloader([m], 1)
                    pka, pka_std = query_model.predict_pka_value(loader)
                    pair = States(
                        pka,
                        pka_std,
                        sorted_mols[0],
                        sorted_mols[1],
                        reaction_center_idx=i,
                        ph7_mol=mol_ph7,
                    )

                    # check if pka is within pH range
                    if pka > max_ph:
                        continue

                    # print(
                    #    "base",
                    #    pka,
                    #    Chem.MolToSmiles(conj),
                    #    i,
                    #    Chem.MolToSmiles(mol_at_state),
                    # )
                    # if bases already present
                    if bases:
                        # check if previous pka is higher
                        if pka > bases[-1].pka:
                            states_per_iteration.append(pair)
                    else:
                        states_per_iteration.append(pair)

                if not states_per_iteration:
                    # no protonation state left
                    break
                # take state with lowest pka value
                bases.append(min(states_per_iteration, key=attrgetter("pka")))
                mol_at_state = deepcopy(bases[-1].deprotonated_mol)
                used_reaction_center_atom_idxs.remove(
                    bases[-1].reaction_center_idx
                )  # avoid double deprotonation

            # print(bases)
            acids.reverse()
            mols = bases + acids
            # print('mols', mols)
            # remove possible duplications
            mols = _check_for_duplicates(mols)

            # if len(mols) == 0:
            #    print("#########################")
            #    print("Could not identify any ionizable group. Aborting.")
            #    print("#########################")

            mol_total += mols
        if not mol_total:
            m = oemol_to_rdmol(mols_ph7[0])
            mol_total = [
                States(
                    7.4,
                    0.0,
                    m,
                    m,
                    reaction_center_idx=-1,
                    ph7_mol=m,
                )
            ]
        # print('title_idx', title_idx, 'mol_total', mol_total)
        yield title_idx, mol_total


def get_neutral_ph_protomer(oemol):
    title = oemol.GetTitle()
    opts = oequacpac.OEMultistatepKaModelOptions()

    multistatepka = oequacpac.OEMultistatepKaModel(oemol, opts)

    ret = []
    if multistatepka.GenerateMicrostates():
        for a in multistatepka.GetMicrostates():
            a.SetTitle(title)
            ret.append(a.CreateCopy())

    ret = remove_highly_charged_mols(ret)
    return ret


def oemol_to_rdmol(oemol):
    """Converts an OpenEye molecule to an RDKit molecule"""
    smi = oechem.OEMolToSmiles(oemol)
    rdmol = Chem.MolFromSmiles(smi)
    return rdmol


def write_enum_smi(output_file, df):
    with open(output_file, "w") as f:
        for i, row in df.iterrows():
            smiles = row["smiles"]
            title = row["title"]
            pka = row["pka"]
            pka_dev = row["pka_dev"]
            print(f"{smiles} {title} {pka} {pka_dev}", file=f)


def write_smi(output_file, df):
    with open(output_file, "w") as f:
        for i, row in df.iterrows():
            smiles = row["smiles"]
            title = row["title"]
            pka_info = row["pka_info"]
            print(f"{smiles} {title} {pka_info}", file=f)


def run(oemols, titles, min_ph, max_ph, enum_mode=False):
    assert min_ph <= 7.4
    assert max_ph >= 7.4

    # predict neutral pH state
    neutral_states = []
    for i, oemol in enumerate(oemols):
        mols = get_neutral_ph_protomer(oemol)
        neutral_states.append(mols)

    if min_ph == 7.4 and max_ph == 7.4:
        # no need to enumerate
        out = []
        for idx in range(len(neutral_states)):
            mols = neutral_states[idx]
            for mol in mols:
                smi = oechem.OEMolToSmiles(mol)
                if enum_mode:
                    out.append(
                        {
                            "idx": idx,
                            "title": titles[idx],
                            "smiles": smi,
                            "pka": None,
                            "pka_dev": None,
                        }
                    )
                else:
                    out.append(
                        {
                            "idx": idx,
                            "title": titles[idx],
                            "smiles": smi,
                            "pka_info": [],
                        }
                    )
        return pd.DataFrame(out)

    # other pKa ranges
    # enumerate neutral protonation states
    protonated_states = enumerate_protonation_states(neutral_states)

    # predict pKa values
    out = []
    if enum_mode:
        seen = set()
        for idx, protonation_states in get_protonation_states(
            neutral_states, protonated_states, min_ph, max_ph
        ):
            pka_info = get_pka_info(protonation_states)
            for c in pka_info:
                center_idx, pka, pka_dev, pro_smi, depro_smi = c.split(";")
                if int(center_idx) < 0:
                    out.append(
                        {
                            "idx": idx,
                            "title": titles[idx],
                            "smiles": oechem.OEMolToSmiles(neutral_states[idx][0]),
                            "pka": None,
                            "pka_dev": None,
                        }
                    )
                    continue
                inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(pro_smi))
                if (idx, inchikey) not in seen:
                    out.append(
                        {
                            "idx": idx,
                            "title": titles[idx],
                            "smiles": pro_smi,
                            "pka": pka,
                            "pka_dev": pka_dev,
                        }
                    )
                    seen.add((idx, inchikey))
                inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(depro_smi))
                if (idx, inchikey) not in seen:
                    out.append(
                        {
                            "idx": idx,
                            "title": titles[idx],
                            "smiles": depro_smi,
                            "pka": pka,
                            "pka_dev": pka_dev,
                        }
                    )
                    seen.add((idx, inchikey))
    else:
        out_dict = {}
        for idx, protonation_states in get_protonation_states(
            neutral_states, protonated_states, min_ph, max_ph
        ):
            if idx not in out_dict:
                out_dict[idx] = {
                    "idx": idx,
                    "title": titles[idx],
                    "smiles": oechem.OEMolToSmiles(oemols[idx]),
                    "pka_info": [],
                }
            for info in get_pka_info(protonation_states):
                c = info.split(";")
                if int(c[0]) < 0:
                    continue
                out_dict[idx]["pka_info"].append(info)
        for idx in out_dict:
            d = out_dict[idx]
            out.append(
                {
                    "idx": idx,
                    "title": d["title"],
                    "smiles": d["smiles"],
                    "pka_info": list(set(d["pka_info"])),
                }
            )
    df = pd.DataFrame(out)

    return df


class ProtonationStateEnumerator(object):
    def __init__(self, low_ph=5.4, high_ph=9.4, max_variants=16, verbose=False):
        self.low_ph = low_ph
        self.high_ph = high_ph
        self.max_variants = max_variants
        assert self.max_variants % 2 == 0
        self.verbose = verbose

    def enum(self, mol):
        mol = mol.CreateCopy()
        title = mol.SetTitle("0")
        df = run([mol], [title], self.low_ph, self.high_ph, enum_mode=True)
        df.sort_values(by=["pka"], inplace=True)
        n = min(self.max_variants, len(df))
        df = df.iloc[:n]
        for smi in df["smiles"]:
            yield smi


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="CSV(w/ smiles,title) or SMI format")
    parser.add_argument("output_file", help="CSV or SMI format (smiles,title,pka_info)")
    parser.add_argument(
        "-e", dest="enum_mode", action="store_true", help="enumeration mode"
    )
    parser.add_argument(
        "-min_ph",
        dest="min_ph",
        type=float,
        default=7.4,
        help="pH lower bound (default: 7.4)",
    )
    parser.add_argument(
        "-max_ph",
        dest="max_ph",
        type=float,
        default=7.4,
        help="pH upper bound (default: 7.4)",
    )
    args = parser.parse_args()

    # check outputfile
    if (
        not args.output_file.endswith(".smi")
        and not args.output_file.endswith(".csv")
        and not args.output_file.endswith(".csv.gz")
    ):
        raise ValueError("Output file must be .smi or .csv or .csv.gz")

    # read input file
    if args.input_file.endswith(".csv") or args.input_file.endswith(".csv.gz"):
        df = pd.read_csv(args.input_file)
        smiles = df["smiles"].tolist()
        titles = df["title"].tolist()
    elif args.input_file.endswith(".smi"):
        smiles = []
        titles = []
        with open(args.input_file) as f:
            for line in f:
                c = line.split()
                smiles.append(c[0])
                try:
                    titles.append(c[1])
                except:
                    titles.append("")
    else:
        raise ValueError("Unknown file format")

    # prepare OEMols
    oemols = []
    for i in range(len(smiles)):
        smi = smiles[i]
        mol = oechem.OEMol()
        oechem.OESmilesToMol(mol, smi)
        mol.SetTitle(str(i))
        oemols.append(mol)

    df = run(oemols, titles, args.min_ph, args.max_ph, enum_mode=args.enum_mode)

    # write
    if args.output_file.endswith(".smi"):
        if args.enum_mode:
            write_enum_smi(args.output_file, df)
        else:
            write_smi(args.output_file, df)
    else:  # csv format
        df.to_csv(args.output_file, index=False)
