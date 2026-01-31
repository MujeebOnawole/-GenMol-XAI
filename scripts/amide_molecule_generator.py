import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Handle both package and standalone execution
try:
    from .utils.smiles_cleaner import clean_smiles
    from .utils.valency_completion import fix_valency
except ImportError:
    try:
        from utils.smiles_cleaner import clean_smiles
        from utils.valency_completion import fix_valency
    except ImportError:
        # Fallback for standalone execution
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils.smiles_cleaner import clean_smiles
        from utils.valency_completion import fix_valency

try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem import QED
    from rdkit.Chem.inchi import MolToInchiKey
    from rdkit.Chem import rdMolDescriptors as rdm
    # SA score
    from rdkit.Chem.SA_Score import sascorer
    # Safety filters (PAINS)
    from rdkit.Chem import FilterCatalog
    from rdkit.Chem import rdchem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    DataStructs = None  # type: ignore
    AllChem = None  # type: ignore
    Descriptors = Crippen = Lipinski = rdMolDescriptors = None  # type: ignore
    QED = None  # type: ignore
    rdm = None  # type: ignore
    MolToInchiKey = None  # type: ignore
    sascorer = None  # type: ignore
    FilterCatalog = None  # type: ignore


REACTION_SMIRKS = "[C:1](=[O:2])[O:3][H].[N:5;H2,H1;!$(N=[!#6])]>>[C:1](=[O:2])[N:5]"
ACID_SMARTS_NATIVE = "[CX3](=O)[OX1H]"
ACID_SMARTS_CARBOXYLATE = "[CX3](=O)[O-]"
ESTER_SMARTS_LATENT = "[CX3](=O)[OX2;!H]"
PRIM_AMINE_SMARTS = "[NX3;H2;!$(NC=O)]"
SEC_AMINE_SMARTS = "[NX3;H1;!$(NC=O)]"
PRIM_AMINE_PROT = "[N+;H3]"
SEC_AMINE_PROT = "[N+;H2]"


@dataclass
class Fragment:
    fragment_id: str
    smiles: str
    role: Optional[str]
    source_csv: str
    reactive_handles: Dict[str, bool]
    safety_is_safe: bool
    safety_alerts: List[str]
    # Training provenance for Tier 3 analysis
    avg_attribution: Optional[float] = None
    activity_rate_percent: Optional[float] = None
    library_type: Optional[str] = None  # SA, EC, CA, SA_EC, SA_CA, CA_EC, TRIPLE


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_library_type_from_source(source_csv: str) -> Optional[str]:
    """Infer library type from source CSV path."""
    src_lower = source_csv.lower()
    if "triple" in src_lower:
        return "TRIPLE"
    elif "dual_sa_ec" in src_lower:
        return "SA_EC"
    elif "dual_sa_ca" in src_lower:
        return "SA_CA"
    elif "dual_ca_ec" in src_lower or "dual_ec_ca" in src_lower:
        return "CA_EC"
    elif "sa_specific" in src_lower:
        return "SA"
    elif "ec_specific" in src_lower:
        return "EC"
    elif "ca_specific" in src_lower:
        return "CA"
    return None


def _load_safe_library(pathogen_dir: str, pathogen: str) -> List[Fragment]:
    """Load safe library for a given library type.

    Library types:
    - SA, EC, CA: Pathogen-specific
    - SA_EC, SA_CA, CA_EC: Dual-active
    - TRIPLE: Triple-active

    Files are named: safe_library_{PATHOGEN}.json

    Extracts training provenance (attribution, activity rate) for Tier 3 analysis.
    """
    # Try new naming convention first, fall back to old
    path = os.path.join(pathogen_dir, f"safe_library_{pathogen}.json")
    if not os.path.exists(path):
        # Fall back to old naming
        path = os.path.join(pathogen_dir, "safe_library.json")
    raw = _read_json(path)
    out: List[Fragment] = []
    for r in raw:
        # Extract provenance fields for training context
        prov_fields = r.get("provenance", {}).get("fields", {})
        source_csv = r.get("provenance", {}).get("source_csv", "")

        # Parse attribution and activity rate
        avg_attr = None
        activity_rate = None
        try:
            avg_attr = float(prov_fields.get("avg_attribution")) if prov_fields.get("avg_attribution") else None
        except (ValueError, TypeError):
            pass
        try:
            # Handle both 'activity_rate_percent' and 'avg_activity_rate_percent'
            rate_val = prov_fields.get("activity_rate_percent") or prov_fields.get("avg_activity_rate_percent")
            activity_rate = float(rate_val) if rate_val else None
        except (ValueError, TypeError):
            pass

        # Infer library type from source CSV
        lib_type = _infer_library_type_from_source(source_csv)

        out.append(
            Fragment(
                fragment_id=r.get("fragment_id"),
                smiles=r.get("fragment_smiles"),
                role=r.get("role"),
                source_csv=source_csv,
                reactive_handles=r.get("reactive_handles", {}),
                safety_is_safe=bool(r.get("safety", {}).get("is_safe", True)),
                safety_alerts=list(r.get("safety", {}).get("alerts", [])),
                avg_attribution=avg_attr,
                activity_rate_percent=activity_rate,
                library_type=lib_type,
            )
        )
    return out


def _compute_props(smiles: str) -> Dict[str, Any]:
    if not Chem:
        return {k: None for k in ["MW", "LogP", "TPSA", "HBA", "HBD", "AromRings", "RotBonds"]}
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return {k: None for k in ["MW", "LogP", "TPSA", "HBA", "HBD", "AromRings", "RotBonds"]}
    return {
        "MW": float(Descriptors.MolWt(m)),
        "LogP": float(Crippen.MolLogP(m)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(m)),
        "HBA": int(Lipinski.NumHAcceptors(m)),
        "HBD": int(Lipinski.NumHDonors(m)),
        "AromRings": int(rdMolDescriptors.CalcNumAromaticRings(m)),
        "RotBonds": int(Lipinski.NumRotatableBonds(m)),
    }


def _compute_qed_sa(smiles: str) -> Tuple[Optional[float], Optional[float]]:
    if not Chem:
        return None, None
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return None, None
    qed_v = None
    sa_v = None
    try:
        if QED is not None:
            qed_v = float(QED.qed(m))
    except Exception:
        qed_v = None
    try:
        if sascorer is not None:
            sa_v = float(sascorer.calculateScore(m))
    except Exception:
        sa_v = None
    return qed_v, sa_v


def _inchikey(smiles: str) -> Optional[str]:
    if not Chem or not MolToInchiKey:
        return None
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return None
    try:
        return MolToInchiKey(m)
    except Exception:
        return None


def _fp_ecfp4(smiles: str):
    if not Chem:
        return None
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _median_tanimoto(smiles_list: List[str]) -> Optional[float]:
    if not Chem or not smiles_list:
        return None
    fps = [ _fp_ecfp4(s) for s in smiles_list[:200] ]  # cap for performance
    fps = [f for f in fps if f is not None]
    if len(fps) < 2:
        return None
    import numpy as np  # type: ignore
    sims = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    if not sims:
        return None
    sims.sort()
    return float(np.median(sims))


def _product_safety_alerts(smiles: str) -> List[str]:
    if not Chem:
        return []
    alerts: List[str] = []
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return ["parse_error"]
    # PAINS A/B using RDKit FilterCatalog if available
    try:
        if FilterCatalog is not None:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
            cat = FilterCatalog.FilterCatalog(params)
            matches = cat.GetMatches(m)
            for match in matches:
                name = match.GetDescription()
                if "PAINS" in name:
                    if "PAINS-A" in name or "PAINS A" in name:
                        alerts.append("PAINS_A")
                    elif "PAINS-B" in name or "PAINS B" in name:
                        alerts.append("PAINS_B")
                    else:
                        alerts.append("PAINS")
    except Exception:
        pass
    # Minimal SMARTS for other hard alerts
    patt_defs = [
        ("michael_acceptor", "C=CC=O"),  # alpha,beta-unsaturated carbonyl (simplified)
        ("alkyl_halide", "[CX4][Cl,Br,I]"),
        ("alkyl_sulfonate", "[SX4](=O)(=O)O[CX4]"),
        ("azide", "N=[N+]=N"),
        ("nitro", "[NX3](=O)=O"),
        ("cumulene", "C=C=C"),
    ]
    for name, smarts in patt_defs:
        try:
            p = Chem.MolFromSmarts(smarts)
            if p and m.HasSubstructMatch(p):
                alerts.append(name)
        except Exception:
            continue
    # Deduplicate
    return sorted(set(alerts))


def _select_main_fragment(smiles: str) -> Tuple[Optional[str], List[str]]:
    if not Chem:
        return smiles, []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return smiles, []
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) <= 1:
            return smiles, []
        amide = Chem.MolFromSmarts("[CX3](=O)N")
        kept = None
        for fm in frags:
            try:
                Chem.SanitizeMol(fm)
            except Exception:
                pass
            if amide and fm.HasSubstructMatch(amide):
                kept = fm
                break
        if kept is None:
            def organic_size(fm: "Chem.Mol") -> Tuple[int, int]:
                has_c = any(a.GetAtomicNum() == 6 for a in fm.GetAtoms())
                return (1 if has_c else 0, fm.GetNumHeavyAtoms())
            kept = sorted(frags, key=organic_size, reverse=True)[0]
        kept_smiles = Chem.MolToSmiles(kept, isomericSmiles=True, canonical=True)
        removed: List[str] = []
        for fm in frags:
            if fm is kept:
                continue
            try:
                Chem.SanitizeMol(fm)
            except Exception:
                pass
            removed.append(Chem.MolToSmiles(fm, isomericSmiles=True, canonical=True))
        return kept_smiles, removed
    except Exception:
        return smiles, []


def _label_leaving_group(smi: str) -> str:
    mapping = {
        "C": "Me",
        "CC": "Et",
        "CCC": "n-Pr",
        "C(C)C": "i-Pr",
        "CCCC": "n-Bu",
        "C(C)CC": "i-Bu",
        "CO": "MeO",
        "CCO": "EtO",
        "Cl": "Cl",
        "Br": "Br",
        "I": "I",
    }
    return mapping.get(smi, smi)


def _enum_acid_sites(mol: "Chem.Mol", allow_latent: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    patterns = [
        (Chem.MolFromSmarts(ACID_SMARTS_NATIVE), "native_acid", "acid"),
        (Chem.MolFromSmarts(ACID_SMARTS_CARBOXYLATE), "carboxylate", "carboxylate"),
    ]
    if allow_latent:
        patterns.append((Chem.MolFromSmarts(ESTER_SMARTS_LATENT), "latent_ester", "latent"))
    seen: set = set()
    for patt, origin, state in patterns:
        if not patt:
            continue
        for match in mol.GetSubstructMatches(patt):
            # Identify carbonyl carbon
            carbon_idx = None
            for idx in match:
                a = mol.GetAtomWithIdx(idx)
                if a.GetAtomicNum() == 6:
                    for nb in a.GetNeighbors():
                        b = mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())
                        if nb.GetAtomicNum() == 8 and b is not None and b.GetBondTypeAsDouble() == 2.0:
                            carbon_idx = a.GetIdx()
                            break
                    if carbon_idx is not None:
                        break
            if carbon_idx is None:
                continue
            if carbon_idx in seen:
                continue
            seen.add(carbon_idx)
            out.append({"idx": carbon_idx, "handle_origin_acid": origin, "acid_state": state})
    # Cap per spec
    return out[:4]


def _enum_amine_sites(mol: "Chem.Mol") -> List[Dict[str, Any]]:
    prim = Chem.MolFromSmarts(PRIM_AMINE_SMARTS)
    sec = Chem.MolFromSmarts(SEC_AMINE_SMARTS)
    prim_p = Chem.MolFromSmarts(PRIM_AMINE_PROT)
    sec_p = Chem.MolFromSmarts(SEC_AMINE_PROT)
    out: List[Dict[str, Any]] = []
    def add_matches(patt, origin, state):
        if not patt:
            return
        for match in mol.GetSubstructMatches(patt):
            for idx in match:
                a = mol.GetAtomWithIdx(idx)
                if a.GetAtomicNum() == 7:
                    out.append({"idx": a.GetIdx(), "handle_origin_amine": origin, "amine_state": state})
                    break
    add_matches(prim, "native_amine", "neutral")
    add_matches(sec, "native_amine", "neutral")
    add_matches(prim_p, "protonated_amine", "protonated")
    add_matches(sec_p, "protonated_amine", "protonated")
    # Unique by idx, keep first origin/state encountered
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for m in out:
        if m["idx"] in seen:
            continue
        seen.add(m["idx"])
        uniq.append(m)
    return uniq[:4]


def _rank_acid_sites(mol: "Chem.Mol", fragment_id: str, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for m in sites:
        idx = m["idx"]
        a = mol.GetAtomWithIdx(idx)
        origin = m.get("handle_origin_acid")
        # Latent subclassification for prioritization
        latent_priority = 0
        if origin == "latent_ester":
            cls, _, _ = _classify_latent_site(mol, idx)
            m["latent_class"] = cls  # B/C/DISALLOWED
            latent_priority = 0 if cls == "B" else 1
        # priority: native(0) < carboxylate(1) < latent_B(2) < latent_other(3)
        if origin == "native_acid":
            pri = 0
        elif origin == "carboxylate":
            pri = 1
        elif origin == "latent_ester":
            pri = 2 + latent_priority
        else:
            pri = 4
        ranked.append((pri, a.IsInRing(), idx, m))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    return [m for _, _, _, m in ranked]


def _rank_amine_sites(mol: "Chem.Mol", fragment_id: str, sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for m in sites:
        idx = m["idx"]
        a = mol.GetAtomWithIdx(idx)
        ranked.append((a.GetDegree(), idx, m))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [m for _, _, m in ranked]


def _find_acid_singlebond_oxygen(mol: "Chem.Mol", c_idx: int) -> Optional[int]:
    """Return the oxygen atom index single-bonded to the carbonyl carbon (OH for acids, O- for carboxylates, OR for esters)."""
    a = mol.GetAtomWithIdx(c_idx)
    for nb in a.GetNeighbors():
        if nb.GetAtomicNum() == 8:
            bond = mol.GetBondBetweenAtoms(c_idx, nb.GetIdx())
            if bond is None:
                continue
            if bond.GetBondTypeAsDouble() == 1.0:
                return nb.GetIdx()
    return None


def _classify_latent_site(mol: "Chem.Mol", c_idx: int) -> Tuple[str, Optional[str], Optional[str]]:
    """Classify latent ester site around carbonyl carbon c_idx.
    Returns (cls, reason, leaving_label):
      - cls in {"B","C","DISALLOWED"}
      - reason: for DISALLOWED, a tag (e.g., michael_acceptor, acyl_sulfonate, anhydride, acyl_fluoride)
      - leaving_label: rough label for leaving group (Me, Et, Bn) when applicable
    """
    if not Chem:
        return "C", None, None
    o_idx = _find_acid_singlebond_oxygen(mol, c_idx)
    if o_idx is None:
        return "C", None, None
    o_atom = mol.GetAtomWithIdx(o_idx)
    # Determine the other neighbor of O (not the carbonyl carbon)
    oc_nb = None
    for nb in o_atom.GetNeighbors():
        if nb.GetIdx() != c_idx:
            oc_nb = nb
            break
    # Quick disallowed checks: acyl sulfonate, anhydride, acyl fluoride
    if oc_nb is not None and oc_nb.GetAtomicNum() == 16:
        return "DISALLOWED", "acyl_sulfonate", None
    # anhydride: O single-bonded to two carbonyl carbons
    if oc_nb is not None and oc_nb.GetAtomicNum() == 6:
        # if oc_nb is carbonyl carbon (has double-bonded O neighbor)
        for nb2 in oc_nb.GetNeighbors():
            b = mol.GetBondBetweenAtoms(oc_nb.GetIdx(), nb2.GetIdx())
            if nb2.GetAtomicNum() == 8 and b is not None and b.GetBondTypeAsDouble() == 2.0:
                # likely an anhydride C(=O)-O-C(=O)
                return "DISALLOWED", "anhydride_like", None
    # αβ-unsaturation near this carbonyl carbon (conjugated carbonyl)
    alpha_cs = [nb for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors() if nb.GetAtomicNum() == 6]
    for alpha in alpha_cs:
        for nb3 in alpha.GetNeighbors():
            if nb3.GetIdx() == c_idx:
                continue
            bond = mol.GetBondBetweenAtoms(alpha.GetIdx(), nb3.GetIdx())
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0 and nb3.GetAtomicNum() == 6:
                return "DISALLOWED", "conjugated_carbonyl", None
    # Leaving group label and whitelist detection
    leaving_label = None
    easy = False
    # Steric congestion heuristic (fused/congested core): carbonyl or alkoxy carbon in ring and secondary/tertiary at O-carbon
    congested = False
    c_atom = mol.GetAtomWithIdx(c_idx)
    if oc_nb is not None and oc_nb.GetAtomicNum() == 6:
        # Methyl: oc_nb heavy neighbors excluding O == 0
        deg_heavy = sum(1 for x in oc_nb.GetNeighbors() if x.GetAtomicNum() > 1)
        deg_ex_O = deg_heavy - 1  # exclude the O
        if deg_ex_O == 0 and oc_nb.GetTotalNumHs() >= 3:
            leaving_label = "Me"
            easy = True
        elif deg_ex_O == 1:
            # Ethyl or benzyl
            other = next(x for x in oc_nb.GetNeighbors() if x.GetIdx() != o_idx)
            if other.GetAtomicNum() == 6:
                # if other is terminal CH3
                deg_heavy_other = sum(1 for x in other.GetNeighbors() if x.GetAtomicNum() > 1)
                if deg_heavy_other == 1 and other.GetTotalNumHs() >= 3:
                    leaving_label = "Et"
                    easy = True
                # benzyl: oc_nb connected to aromatic carbon
                if other.GetIsAromatic():
                    leaving_label = "Bn"
                    easy = True
        elif deg_ex_O == 2:
            # Secondary alkyl (e.g., i-Pr, sec-alkyl): non-conjugated and not congested
            # Determine congestion: ring involvement at c_atom or oc_nb
            if oc_nb.IsInRing() or c_atom.IsInRing():
                congested = True
            # i-Pr detection: both non-O neighbors methyl-like
            nbrs = [x for x in oc_nb.GetNeighbors() if x.GetIdx() != o_idx]
            if len(nbrs) == 2 and all(n.GetAtomicNum() == 6 for n in nbrs):
                methylish = 0
                for n in nbrs:
                    deg_hvy_n = sum(1 for x in n.GetNeighbors() if x.GetAtomicNum() > 1)
                    if deg_hvy_n == 1 and n.GetTotalNumHs() >= 2:
                        methylish += 1
                if methylish == 2:
                    leaving_label = "iPr"
                else:
                    leaving_label = "sec-alkyl"
            else:
                leaving_label = "sec-alkyl"
            if not congested:
                easy = True
        elif deg_ex_O == 3:
            # tert-butyl like
            leaving_label = "tBu"
            # allow as B if non-congested and non-conjugated
            if not congested:
                easy = True
            else:
                return "C", "steric_congestion", leaving_label
    # Default classification
    # Heuristic deactivation (e.g., CF3 on ring near carbonyl)
    try:
        cf3 = Chem.MolFromSmarts("[CX4](F)(F)F")
        if cf3:
            c_atom = mol.GetAtomWithIdx(c_idx)
            ring_neighbors = [n for n in c_atom.GetNeighbors() if n.GetIsAromatic()]
            if ring_neighbors:
                # if any CF3 within 2 bonds on the aromatic ring
                for rn in ring_neighbors:
                    for nb in rn.GetNeighbors():
                        if nb.GetIdx() == c_idx:
                            continue
                        sub = mol.GetSubstructMatch(cf3)
                        if sub:
                            return "C", "deactivated_ring", leaving_label
    except Exception:
        pass
    if easy and not congested:
        return "B", f"whitelist:{leaving_label or 'alkyl'}", leaving_label
    return "C", ("steric_congestion" if congested else None), leaving_label


def _force_amide_coupling(acid_smiles: str, acid_c_idx: int, amine_smiles: str, amine_n_idx: int) -> Optional[Tuple[str, int, int]]:
    """Force coupling exactly between the specified acid carbonyl carbon and amine nitrogen.
    Implementation: remove acid OH oxygen, then form a C-N bond. Track product indices by AtomMapNum.
    Returns (product_smiles, product_C_idx, product_N_idx) or None if failed."""
    if not Chem:
        return None
    am = Chem.MolFromSmiles(acid_smiles)
    bm = Chem.MolFromSmiles(amine_smiles)
    if am is None or bm is None:
        return None
    if acid_c_idx >= am.GetNumAtoms() or amine_n_idx >= bm.GetNumAtoms():
        return None
    # Validate acid C has a double-bond O neighbor and an OH neighbor
    c_atom = am.GetAtomWithIdx(acid_c_idx)
    has_c_dbo = any(
        (nb.GetAtomicNum() == 8 and am.GetBondBetweenAtoms(acid_c_idx, nb.GetIdx()) is not None and am.GetBondBetweenAtoms(acid_c_idx, nb.GetIdx()).GetBondTypeAsDouble() == 2.0)
        for nb in c_atom.GetNeighbors()
    )
    sb_o_idx = _find_acid_singlebond_oxygen(am, acid_c_idx)
    if not has_c_dbo or sb_o_idx is None:
        return None
    n_atom = bm.GetAtomWithIdx(amine_n_idx)
    if n_atom.GetAtomicNum() != 7:
        return None
    # Combine molecules
    combo = Chem.CombineMols(am, bm)
    rw = Chem.RWMol(combo)
    amine_off = am.GetNumAtoms()
    c_idx = acid_c_idx
    n_idx = amine_off + amine_n_idx
    # Tag atoms for later lookup after deletions
    rw.GetAtomWithIdx(c_idx).SetAtomMapNum(1)
    rw.GetAtomWithIdx(n_idx).SetAtomMapNum(5)
    # Remove the OH oxygen atom attached to carbon (identify in combined indices)
    oh_combined = sb_o_idx
    # Remove oxygen atom (hydrogen is implicit)
    try:
        # Remove bond C-OH if exists (will be dropped with atom removal too)
        if rw.GetBondBetweenAtoms(c_idx, oh_combined):
            rw.RemoveBond(c_idx, oh_combined)
        # Remove the OH oxygen atom (use higher index removes first strategy not needed here)
        rw.RemoveAtom(oh_combined)
    except Exception:
        return None
    # After removal, indices shift. Find C and N again by AtomMapNum
    c_idx_new = n_idx_new = None
    for atom in rw.GetAtoms():
        amap = atom.GetAtomMapNum()
        if amap == 1:
            c_idx_new = atom.GetIdx()
        elif amap == 5:
            n_idx_new = atom.GetIdx()
    if c_idx_new is None or n_idx_new is None:
        return None
    # Add C-N single bond
    try:
        rw.AddBond(int(c_idx_new), int(n_idx_new), order=Chem.BondType.SINGLE)
    except Exception:
        return None
    try:
        prod = rw.GetMol()
        Chem.SanitizeMol(prod)
        smi = Chem.MolToSmiles(prod, isomericSmiles=True, canonical=True)
    except Exception:
        return None
    # Retrieve final indices for C and N by AtomMapNum
    pc = pn = None
    for atom in prod.GetAtoms():
        if atom.GetAtomMapNum() == 1:
            pc = atom.GetIdx()
        elif atom.GetAtomMapNum() == 5:
            pn = atom.GetIdx()
    if pc is None or pn is None:
        # Fallback: not fatal; but should not happen if tags preserved
        pc = 0
        pn = 0
    return smi, int(pc), int(pn)


def _load_smiles_set(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        return []
    smiles = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ["SMILES", "smiles", "fragment_smiles", "canonical_smiles"]:
                if k in row and row[k]:
                    smiles.append(row[k])
                    break
    return smiles


def _make_dirs(base: str, pathogen: str) -> Dict[str, str]:
    pdir = os.path.join(base, pathogen)
    out = {
        "pathogen_dir": pdir,
        "outputs": os.path.join(pdir, "outputs"),
        "logs": os.path.join(pdir, "logs"),
        "reports": os.path.join(pdir, "reports"),
    }
    for k, v in out.items():
        if k == "pathogen_dir":
            continue
        os.makedirs(v, exist_ok=True)
    return out


def _design_score(props: Dict[str, Any], pathogen: str, fragment_scores: Tuple[float, float] = (0.0, 0.0)) -> float:
    # Deterministic heuristic with soft windows per pathogen
    mw = props.get("MW") or 0.0
    tpsa = props.get("TPSA") or 0.0
    logp = props.get("LogP") or 0.0
    f1, f2 = fragment_scores
    # Base score around moderate ranges
    score = 0.0
    score += max(0.0, 1.0 - abs(mw - 400.0) / 400.0)
    score += max(0.0, 1.0 - abs(tpsa - 75.0) / 75.0)
    score += max(0.0, 1.0 - abs(logp - 2.0) / 2.0)
    # Soft window penalty
    if pathogen == "EC":
        mw_max, lp_max, tpsa_max, hbd_max, rot_max = 550, 5, 120, 3, 8
    else:  # SA, CA, TRIPLE
        mw_max, lp_max, tpsa_max, hbd_max, rot_max = 600, 6, 140, 4, 10
    hbd = props.get("HBD") or 0
    rotb = props.get("RotBonds") or 0
    penalties = 0.0
    for val, vmax in [(mw, mw_max), (logp, lp_max), (tpsa, tpsa_max)]:
        if val > vmax:
            penalties += min(1.0, (val - vmax) / max(1.0, vmax))
    if hbd > hbd_max:
        penalties += min(1.0, (hbd - hbd_max) / max(1.0, hbd_max))
    if rotb > rot_max:
        penalties += min(1.0, (rotb - rot_max) / max(1.0, rot_max))
    score -= 0.5 * penalties
    # Fragment extras placeholder
    score += 0.25 * (f1 + f2)
    return float(round(score, 6))


def generate_products(
    pathogen: str,
    base_outdir: str,
    sa_train: str,
    ec_train: str,
    ca_train: str,
    max_acids: Optional[int],
    max_amines: Optional[int],
    allow_latent_acids: bool,
    pair_batch_size: Optional[int],
    target_count: Optional[int],
    max_products_per_acid: Optional[int],
    score_with_model: bool,
    write_report: bool,
    seed: int,
) -> None:
    if not Chem or not AllChem:
        raise RuntimeError("RDKit not available: generation requires RDKit")

    paths = _make_dirs(base_outdir, pathogen)
    pathogen_dir = paths["pathogen_dir"]

    # Logging setup
    log_path = os.path.join(paths["logs"], f"{pathogen}_generation.log")
    start_time = datetime.utcnow()
    rnd = random.Random(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    with open(log_path, "w", encoding="utf-8") as log:
        def logw(msg: str) -> None:
            log.write(msg + "\n")
        # Header
        pyver = sys.version.split(" ")[0]
        rdkitver = getattr(Chem, "rdkitVersion", "unknown")
        logw(f"Start: {start_time.isoformat()}")
        logw(f"Seed: {seed}")
        logw(f"Python: {pyver} | RDKit: {rdkitver}")
        # git hash optional
        try:
            import subprocess
            gh = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            gh = "unknown"
        logw(f"Git: {gh}")

        # Load libraries
        lib_path = os.path.join(pathogen_dir, "safe_library.json")
        library = _read_json(lib_path)
        fragments: List[Fragment] = _load_safe_library(pathogen_dir, pathogen)
        # Select acids/amines
        acids = [f for f in fragments if f.reactive_handles.get("carboxylic_acid", False)]
        # Latent acids policy: CA and CA-containing dual-actives default ON; if scarcity (<23%) then enable for others
        if pathogen in ("CA", "SA_CA", "CA_EC") and not allow_latent_acids:
            allow_latent_acids = True
        acid_ratio = (len(acids) / max(1, len(fragments))) if fragments else 0.0
        # Auto-enable latent more conservatively (3%) for SA/EC/TRIPLE and their dual-actives
        if not allow_latent_acids and pathogen in ("SA", "EC", "SA_EC", "TRIPLE") and acid_ratio < 0.03:
            logw(f"Latent acids enabled due to scarcity: acids_ratio={acid_ratio:.3f}")
            allow_latent_acids = True
        if allow_latent_acids and Chem is not None:
            ester_pat = Chem.MolFromSmarts(ESTER_SMARTS_LATENT)
            extra_acids: List[Fragment] = []
            for f in fragments:
                if f in acids:
                    continue
                m = Chem.MolFromSmiles(f.smiles)
                if not m or not ester_pat:
                    continue
                if m.HasSubstructMatch(ester_pat):
                    extra_acids.append(f)
            acids.extend(extra_acids)
        amines = [f for f in fragments if f.reactive_handles.get("primary_amine", False) or f.reactive_handles.get("secondary_amine", False)]

        # Optional external acid panel when native+carboxylate acids are scarce (<20)
        external_acid_ids: set = set()
        external_acid_products = 0
        try:
            if Chem is not None:
                patt_na = Chem.MolFromSmarts(ACID_SMARTS_NATIVE)
                patt_cb = Chem.MolFromSmarts(ACID_SMARTS_CARBOXYLATE)
                native_or_carbox = 0
                for f in fragments:
                    m = Chem.MolFromSmiles(f.smiles)
                    if not m:
                        continue
                    if (patt_na and m.HasSubstructMatch(patt_na)) or (patt_cb and m.HasSubstructMatch(patt_cb)):
                        native_or_carbox += 1
                need_external = native_or_carbox < 20
            else:
                need_external = False
        except Exception:
            need_external = False
        ext_panel_path = os.path.join("FBDD", "common", "allowed_external_acid_panel.csv")
        if need_external and os.path.exists(ext_panel_path):
            try:
                with open(ext_panel_path, "r", encoding="utf-8", newline="") as ef:
                    rdr = csv.DictReader(ef)
                    for row in rdr:
                        smi = (row.get("acid_smiles") or "").strip()
                        aid = (row.get("acid_id") or "").strip() or f"ext-{len(external_acid_ids)+1}"
                        if not smi:
                            continue
                        # Build synthetic fragment entry for acids list only
                        acids.append(Fragment(
                            fragment_id=aid,
                            smiles=smi,
                            role="scaffold",
                            source_csv="external_panel",
                            reactive_handles={"carboxylic_acid": True, "primary_amine": False, "secondary_amine": False},
                            safety_is_safe=True,
                            safety_alerts=[],
                        ))
                        external_acid_ids.add(aid)
                logw(f"External acid panel used: {len(external_acid_ids)} entries loaded (cap contribution to 20 products)")
            except Exception:
                logw("External acid panel present but failed to load; continuing without it.")

        # Role-preferential ordering: acids -> scaffold first; amines -> substituent first
        acids.sort(key=lambda x: (0 if x.role == "scaffold" else 1, x.fragment_id, x.smiles))
        amines.sort(key=lambda x: (0 if x.role == "substituent" else 1, x.fragment_id, x.smiles))

        if max_acids is not None:
            acids = acids[:max_acids]
        if max_amines is not None:
            amines = amines[:max_amines]

        logw("Inputs:")
        logw(f"  Safe library: {lib_path}")
        logw(f"  Fragments total: {len(fragments)} | acids: {len(acids)} | amines: {len(amines)}")

        # Novelty sets
        sa_smiles = _load_smiles_set(sa_train)
        ec_smiles = _load_smiles_set(ec_train)
        ca_smiles = _load_smiles_set(ca_train)
        # Decide if InChIKey is available; fallback to canonical SMILES hashing if not
        inchi_available = _inchikey("CC") is not None
        logw(f"InChIKey available: {inchi_available}")
        def to_keys(smis: Sequence[str]) -> List[str]:
            out = []
            for s in smis:
                if not s:
                    continue
                if inchi_available:
                    k = _inchikey(s)
                    if k:
                        out.append(k)
                else:
                    # canonical smiles fallback
                    try:
                        m = Chem.MolFromSmiles(s)
                        if m:
                            out.append(Chem.MolToSmiles(m, isomericSmiles=True, canonical=True))
                    except Exception:
                        continue
            return out
        sa_keys = set(to_keys(sa_smiles))
        ec_keys = set(to_keys(ec_smiles))
        ca_keys = set(to_keys(ca_smiles))
        union_keys = sa_keys | ec_keys | ca_keys
        logw("Novelty sets:")
        logw(f"  SA train: {len(sa_keys)} | EC train: {len(ec_keys)} | CA train: {len(ca_keys)} | union: {len(union_keys)}")

        # Reaction parsing skipped; we record the SMIRKS verbatim for traceability.
        rxn = None

        attempts = successes = site_failures = 0
        valency_fixes = cleanings = 0
        safety_excluded = novelty_excluded = 0
        novelty_excluded_pathogen = 0
        novelty_excluded_union = 0
        kept: List[List[Any]] = []
        seen_keys: set = set()
        removed_duplicates = 0
        alt_rationales_by_key: Dict[str, str] = {}
        failure_rows: List[List[Any]] = []
        # Safety alert class counts
        safety_alert_counts: Dict[str, int] = {}
        # Handle-origin counters
        c_native = c_carbox = c_latent = 0
        n_neutral = n_prot = 0
        pair_counter = 0
        stop_early = False
        # Per-acid fan-out cap
        acid_products: Dict[str, int] = {}
        # TRIPLE pre-join safety skips
        prejoin_safety_skipped = 0
        # Latent disallowed pre-join skips
        prejoin_latent_disallowed = 0
        prejoin_latent_motif_counts: Dict[str, int] = {}
        # EC-only predicted Michael acceptor pre-join veto
        prejoin_predicted_michael_acceptor = 0
        # Side fragments audit
        side_frag_total = 0
        side_frag_examples: List[str] = []
        latent_used_count = 0
        leaving_groups_counts: Dict[str, int] = {}
        # Route reason breakdown (B vs C reasons)
        route_reason_counts: Dict[str, int] = {}

        # Optional model telemetry stub
        if score_with_model:
            logw("Model telemetry: --score-with-model passed; no predictor integrated (predict.py not found). Skipping model scoring.")

        # Enforce EC-specific fan-out cap (max 25)
        if pathogen == "EC" and max_products_per_acid is not None and max_products_per_acid > 25:
            max_products_per_acid = 25

        for a in acids:
            if max_products_per_acid is not None and acid_products.get(a.fragment_id, 0) >= max_products_per_acid:
                continue
            # External panel contribution cap (<=20)
            is_external_acid = a.fragment_id in external_acid_ids or a.source_csv == "external_panel"
            if is_external_acid and external_acid_products >= 20:
                continue
            mol_a = Chem.MolFromSmiles(a.smiles)
            if not mol_a:
                continue
            acid_candidates = _enum_acid_sites(mol_a, allow_latent_acids)
            # Count sites by origin
            for m in acid_candidates:
                if m["handle_origin_acid"] == "native_acid":
                    c_native += 1
                elif m["handle_origin_acid"] == "carboxylate":
                    c_carbox += 1
                elif m["handle_origin_acid"] == "latent_ester":
                    c_latent += 1
            a_sites = _rank_acid_sites(mol_a, a.fragment_id, acid_candidates)
            for b in amines:
                if stop_early:
                    break
                if pair_batch_size is not None and pair_counter >= pair_batch_size:
                    break
                if max_products_per_acid is not None and acid_products.get(a.fragment_id, 0) >= max_products_per_acid:
                    break
                # Role preference already applied via ordering
                mol_b = Chem.MolFromSmiles(b.smiles)
                if not mol_b:
                    continue
                # TRIPLE-only pre-join safety: skip unsafe pairs before attempting coupling
                if pathogen == "TRIPLE":
                    a_alerts = _product_safety_alerts(a.smiles)
                    b_alerts = _product_safety_alerts(b.smiles)
                    if a_alerts or b_alerts:
                        prejoin_safety_skipped += 1
                        continue
                amine_candidates = _enum_amine_sites(mol_b)
                for m in amine_candidates:
                    if m["amine_state"] == "neutral":
                        n_neutral += 1
                    elif m["amine_state"] == "protonated":
                        n_prot += 1
                b_sites = _rank_amine_sites(mol_b, b.fragment_id, amine_candidates)
                if not a_sites or not b_sites:
                    continue
                # Prepare alternatives list (top 3 rationale)
                alt_sites = []
                # Build ordered attempt list for strict site forcing (max 3 attempts)
                ordered_acids = a_sites[:3]
                ordered_amines = b_sites[:3]
                attempt_pairs: List[Tuple[int, int]] = []
                if ordered_acids and ordered_amines:
                    attempt_pairs.append((ordered_acids[0]["idx"], ordered_amines[0]["idx"]))
                    if len(ordered_amines) > 1:
                        attempt_pairs.append((ordered_acids[0]["idx"], ordered_amines[1]["idx"]))
                    if len(ordered_acids) > 1:
                        attempt_pairs.append((ordered_acids[1]["idx"], ordered_amines[0]["idx"]))

                prod_smiles = None
                prod_c_idx = prod_n_idx = None
                chosen_acid = chosen_amine = None
                for attempt_i, (acid_idx, amine_idx) in enumerate(attempt_pairs, start=1):
                    attempts += 1
                    pair_counter += 1
                    # Pre-filter disallowed latent motifs for this acid site
                    chosen_site_meta = next((m for m in a_sites if m["idx"] == acid_idx), None)
                    if chosen_site_meta and chosen_site_meta.get("handle_origin_acid") == "latent_ester":
                        cls, reason, _ = _classify_latent_site(mol_a, acid_idx)
                        if cls == "DISALLOWED":
                            prejoin_latent_disallowed += 1
                            if reason:
                                prejoin_latent_motif_counts[reason] = prejoin_latent_motif_counts.get(reason, 0) + 1
                            # log failure attempt
                            failure_rows.append([
                                pathogen,
                                a.fragment_id,
                                a.smiles,
                                b.fragment_id,
                                b.smiles,
                                attempt_i,
                                acid_idx,
                                amine_idx,
                                "prejoin",
                                "latent_disallowed_motif",
                                (len(a_sites)-1 + len(b_sites)-1),
                                json.dumps(alt_sites[:3]),
                                datetime.utcnow().isoformat(),
                            ])
                            continue
                    # EC-only predicted Michael acceptor veto even for native/carboxylate
                    if pathogen == "EC":
                        cls2, reason2, _ = _classify_latent_site(mol_a, acid_idx)
                        if cls2 == "DISALLOWED" and reason2 == "conjugated_carbonyl":
                            prejoin_predicted_michael_acceptor += 1
                            failure_rows.append([
                                pathogen,
                                a.fragment_id,
                                a.smiles,
                                b.fragment_id,
                                b.smiles,
                                attempt_i,
                                acid_idx,
                                amine_idx,
                                "prejoin",
                                "michael_acceptor_precursor",
                                (len(a_sites)-1 + len(b_sites)-1),
                                json.dumps(alt_sites[:3]),
                                datetime.utcnow().isoformat(),
                            ])
                            continue
                    forced = _force_amide_coupling(a.smiles, acid_idx, b.smiles, amine_idx)
                    if forced is not None:
                        prod_smiles, prod_c_idx, prod_n_idx = forced
                        chosen_acid, chosen_amine = acid_idx, amine_idx
                        break
                    else:
                        # record rationale
                        if acid_idx != attempt_pairs[0][0]:
                            reason = "non-ring prioritized" if mol_a.GetAtomWithIdx(a_sites[0]).IsInRing() and not mol_a.GetAtomWithIdx(acid_idx).IsInRing() else "atom_idx order"
                            alt_sites.append({"acid_atom": acid_idx, "reason": reason})
                        if amine_idx != attempt_pairs[0][1]:
                            reason = "lower degree prioritized" if mol_b.GetAtomWithIdx(b_sites[0]).GetDegree() > mol_b.GetAtomWithIdx(amine_idx).GetDegree() else "atom_idx order"
                            alt_sites.append({"amine_atom": amine_idx, "reason": reason})
                        # log failure attempt
                        failure_rows.append([
                            pathogen,
                            a.fragment_id,
                            a.smiles,
                            b.fragment_id,
                            b.smiles,
                            attempt_i,
                            acid_idx,
                            amine_idx,
                            "site_forcing",
                            "force_failed",
                            (len(a_sites)-1 + len(b_sites)-1),
                            json.dumps(alt_sites[:3]),
                            datetime.utcnow().isoformat(),
                        ])
                if prod_smiles is None:
                    site_failures += 1
                    continue
                alt_json = json.dumps(alt_sites[:3])
                val_fixed_flag = False
                cleaned_flag = False
                # Valency â†’ cleaning
                fixed, vf = fix_valency(prod_smiles)
                if fixed:
                    prod_smiles = fixed
                    valency_fixes += int(vf)
                    val_fixed_flag = vf
                else:
                    # Record failure and skip product
                    failure_rows.append([
                        pathogen,
                        a.fragment_id,
                        a.smiles,
                        b.fragment_id,
                        b.smiles,
                        1,
                        chosen_acid,
                        chosen_amine,
                        "valency",
                        "valency_fix_failed",
                        (len(a_sites)-1 + len(b_sites)-1),
                        alt_json,
                        datetime.utcnow().isoformat(),
                    ])
                    continue
                cleaned, cf = clean_smiles(prod_smiles)
                if cleaned:
                    prod_smiles = cleaned
                    cleanings += int(cf)
                    cleaned_flag = cf
                else:
                    # Record failure and skip product
                    failure_rows.append([
                        pathogen,
                        a.fragment_id,
                        a.smiles,
                        b.fragment_id,
                        b.smiles,
                        1,
                        chosen_acid,
                        chosen_amine,
                        "cleaning",
                        "smiles_clean_failed",
                        (len(a_sites)-1 + len(b_sites)-1),
                        alt_json,
                        datetime.utcnow().isoformat(),
                    ])
                    continue
                # Properties & identity
                # Single-component policy: keep amide-containing fragment (or largest organic)
                removed_frags: List[str] = []
                main_smi, removed = _select_main_fragment(prod_smiles)
                if main_smi and main_smi != prod_smiles:
                    prod_smiles = main_smi
                    removed_frags = removed
                    side_frag_total += len(removed_frags)
                    if removed_frags:
                        side_frag_examples.extend(removed_frags[:3])
                # Identity key (InChI or SMILES fallback)
                inchikey = _inchikey(prod_smiles) if inchi_available else None
                if not inchikey:
                    try:
                        mtmp = Chem.MolFromSmiles(prod_smiles)
                        inchikey = Chem.MolToSmiles(mtmp, isomericSmiles=True, canonical=True) if mtmp else ""
                    except Exception:
                        inchikey = ""
                props = _compute_props(prod_smiles)
                # Handle origins/states
                chosen_acid_meta = next((m for m in a_sites if m["idx"] == chosen_acid), {"handle_origin_acid": "", "acid_state": ""})
                chosen_amine_meta = next((m for m in b_sites if m["idx"] == chosen_amine), {"handle_origin_amine": "", "amine_state": ""})
                # Safety (fragment-level and product-level alerts)
                safety_alerts: List[str] = []
                product_alerts = _product_safety_alerts(prod_smiles)
                for al in product_alerts:
                    safety_alert_counts[al] = safety_alert_counts.get(al, 0) + 1
                safety_is_safe = a.safety_is_safe and b.safety_is_safe and (len(product_alerts) == 0)
                # aggregate alerts for traceability
                if product_alerts:
                    safety_alerts.extend(product_alerts)
                if not a.safety_is_safe:
                    safety_alerts.extend([f"acid:{al}" for al in (a.safety_alerts or ["unsafe_fragment"])])
                if not b.safety_is_safe:
                    safety_alerts.extend([f"amine:{al}" for al in (b.safety_alerts or ["unsafe_fragment"])])
                # Route classification (A/B/C)
                route_class = "A"
                if chosen_acid_meta.get("handle_origin_acid") == "native_acid" or chosen_acid_meta.get("handle_origin_acid") == "carboxylate":
                    route_class = "A"
                elif chosen_acid_meta.get("handle_origin_acid") == "latent_ester":
                    cls_lat, reason_lat, leaving = _classify_latent_site(mol_a, int(chosen_acid))
                    route_class = "B" if cls_lat == "B" else "C"
                    key = reason_lat or (f"whitelist:{leaving or 'alkyl'}" if cls_lat == 'B' else 'other')
                    route_reason_counts[key] = route_reason_counts.get(key, 0) + 1
                # Novelty - check against relevant training sets based on library type
                if pathogen == "SA":
                    novel_vs_pathogen = inchikey not in sa_keys if inchikey else True
                elif pathogen == "EC":
                    novel_vs_pathogen = inchikey not in ec_keys if inchikey else True
                elif pathogen == "CA":
                    novel_vs_pathogen = inchikey not in ca_keys if inchikey else True
                elif pathogen == "SA_EC":
                    # Dual-active: check against both SA and EC
                    novel_vs_pathogen = (inchikey not in sa_keys and inchikey not in ec_keys) if inchikey else True
                elif pathogen == "SA_CA":
                    # Dual-active: check against both SA and CA
                    novel_vs_pathogen = (inchikey not in sa_keys and inchikey not in ca_keys) if inchikey else True
                elif pathogen == "CA_EC":
                    # Dual-active: check against both CA and EC
                    novel_vs_pathogen = (inchikey not in ca_keys and inchikey not in ec_keys) if inchikey else True
                else:  # TRIPLE
                    novel_vs_pathogen = inchikey not in union_keys if inchikey else True
                novel_vs_union = inchikey not in union_keys if inchikey else True
                # Safety hard filter
                if not safety_is_safe:
                    safety_excluded += 1
                    failure_rows.append([
                        pathogen,
                        a.fragment_id,
                        a.smiles,
                        b.fragment_id,
                        b.smiles,
                        1,
                        chosen_acid,
                        chosen_amine,
                        "safety",
                        ";".join(safety_alerts) or "safety_filter",
                        (len(a_sites)-1 + len(b_sites)-1),
                        alt_json,
                        datetime.utcnow().isoformat(),
                    ])
                    continue

                # Novelty filter and de-dup incrementally
                if not (novel_vs_pathogen and novel_vs_union):
                    novelty_excluded += 1
                    if not novel_vs_pathogen:
                        novelty_excluded_pathogen += 1
                    if not novel_vs_union:
                        novelty_excluded_union += 1
                    continue
                if inchikey and inchikey in seen_keys:
                    removed_duplicates += 1
                    continue
                if inchikey:
                    seen_keys.add(inchikey)
                filtered_reason = ""
                # Scoring
                # QED / SA score
                qed_v, sa_v = _compute_qed_sa(prod_smiles)
                # Scoring
                dscore = _design_score(props, pathogen)
                successes += 1
                # Latent telemetry (count and leaving groups)
                if chosen_acid_meta.get("handle_origin_acid") == "latent_ester":
                    latent_used_count += 1
                    for frag_smi in removed_frags:
                        label = _label_leaving_group(frag_smi)
                        leaving_groups_counts[label] = leaving_groups_counts.get(label, 0) + 1
                row = [
                    prod_smiles,
                    inchikey,
                    pathogen,
                    datetime.utcnow().isoformat(),
                    "amide",
                    REACTION_SMIRKS,
                    a.fragment_id,
                    a.smiles,
                    a.source_csv,
                    a.role or "unknown",
                    True if is_external_acid else False,
                    a.library_type or "",  # NEW: acid library type
                    a.avg_attribution,  # NEW: acid training attribution
                    a.activity_rate_percent,  # NEW: acid training activity rate
                    b.fragment_id,
                    b.smiles,
                    b.source_csv,
                    b.role or "unknown",
                    b.library_type or "",  # NEW: amine library type
                    b.avg_attribution,  # NEW: amine training attribution
                    b.activity_rate_percent,  # NEW: amine training activity rate
                    f"C_idx={prod_c_idx};N_idx={prod_n_idx}",
                    str(chosen_acid),
                    str(chosen_amine),
                    len(a_sites) - 1 + len(b_sites) - 1,
                    alt_json,
                    chosen_acid_meta.get("handle_origin_acid"),
                    chosen_acid_meta.get("acid_state"),
                    chosen_amine_meta.get("handle_origin_amine"),
                    chosen_amine_meta.get("amine_state"),
                    "ester_aminolysis" if chosen_acid_meta.get("handle_origin_acid") == "latent_ester" else "",
                    route_class,
                    bool(val_fixed_flag),
                    bool(cleaned_flag),
                    bool(safety_is_safe),
                    ";".join(safety_alerts),
                    bool(novel_vs_pathogen),
                    bool(novel_vs_union),
                    filtered_reason,
                    props.get("MW"),
                    props.get("LogP"),
                    props.get("TPSA"),
                    props.get("HBA"),
                    props.get("HBD"),
                    props.get("AromRings"),
                    props.get("RotBonds"),
                    qed_v,  # QED
                    sa_v,   # SA_score
                    dscore,
                    None,  # rank_within_pathogen filled later
                    (len(removed_frags) if removed_frags else 0),
                ]
                kept.append(row)
                # Per-acid fan-out counter
                acid_products[a.fragment_id] = acid_products.get(a.fragment_id, 0) + 1
                if is_external_acid:
                    external_acid_products += 1
                # Early stop by target count
                if target_count is not None and len(kept) >= target_count:
                    stop_early = True
                    break
            if stop_early:
                break
        # End loops; proceed to ranking and writing outputs

        # Rank by design_score (third from end)
        kept.sort(key=lambda x: float(x[-3]) if x[-3] is not None else 0.0, reverse=True)
        for i, r in enumerate(kept, start=1):
            r[-2] = i

        # Write outputs
        out_smi = os.path.join(paths["outputs"], f"{pathogen}_amide_library.smi")
        out_csv = os.path.join(paths["outputs"], f"{pathogen}_amide_library_detailed.csv")
        out_fail = os.path.join(paths["outputs"], f"{pathogen}_amide_generation_failures.csv")
        with open(out_smi, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(r[0] + "\n")
        headers = [
            "product_smiles","product_inchikey","pathogen","generation_timestamp","reaction_type","rxn_smarts",
            "acid_fragment_id","acid_fragment_smiles","acid_source_csv","acid_role","acid_external_panel",
            "acid_library_type","acid_avg_attribution","acid_activity_rate",  # NEW: Training provenance
            "amine_fragment_id","amine_fragment_smiles","amine_source_csv","amine_role",
            "amine_library_type","amine_avg_attribution","amine_activity_rate",  # NEW: Training provenance
            "coupling_atoms_product","coupling_atoms_acid_fragment","coupling_atoms_amine_fragment",
            "alternative_sites_considered","alternative_sites_summary",
            "handle_origin_acid","acid_state","handle_origin_amine","amine_state","handle_exposure_method","route_class",
            "valency_fixed","cleaning_applied","safety_is_safe","safety_alerts","novel_vs_pathogen_train","novel_vs_union_train","filtered_reason",
            "MW","LogP","TPSA","HBA","HBD","AromRings","RotBonds","QED","SA_score","design_score","rank_within_pathogen","side_fragments_removed"
        ]
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in kept:
                writer.writerow(r)
        # Failures CSV
        fail_headers = [
            "pathogen","acid_fragment_id","acid_fragment_smiles","amine_fragment_id","amine_fragment_smiles",
            "attempt_order","selected_acid_atom_idx","selected_amine_atom_idx","failure_stage","failure_reason",
            "alternative_sites_considered","alternative_sites_summary","timestamp"
        ]
        with open(out_fail, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(fail_headers)
            for row in failure_rows:
                w.writerow(row)

        # Diversity metric
        median_div = _median_tanimoto([r[0] for r in kept]) if kept else None
        # Write A/B/C filtered CSVs (successes only)
        out_csv = os.path.join(paths["outputs"], f"{pathogen}_amide_library_detailed.csv")
        def _write_filtered(route_label: str, fname: str) -> None:
            p = os.path.join(paths["outputs"], fname)
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(headers)
                for r in kept:
                    try:
                        rc_idx = headers.index("route_class")
                    except ValueError:
                        continue
                    if str(r[rc_idx]) == route_label:
                        w.writerow(r)
        _write_filtered("A", f"{pathogen}_amide_A_direct.csv")
        _write_filtered("B", f"{pathogen}_amide_B_easy_latent.csv")
        _write_filtered("C", f"{pathogen}_amide_C_hard_latent.csv")

        # Diverse40 selection primarily from A/B
        def _greedy_maxmin(smiles_list: List[str], k: int = 40) -> List[int]:
            if not Chem or not smiles_list:
                return list(range(min(k, len(smiles_list))))
            fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles_list]
            idxs: List[int] = []
            if not fps:
                return idxs
            idxs.append(0)
            while len(idxs) < min(k, len(fps)):
                best_j = None
                best_min_sim = 2.0
                for j in range(len(fps)):
                    if j in idxs:
                        continue
                    sims = [DataStructs.TanimotoSimilarity(fps[j], fps[i]) for i in idxs]
                    min_sim = min(sims) if sims else 0.0
                    if min_sim < best_min_sim:
                        best_min_sim = min_sim
                        best_j = j
                if best_j is None:
                    break
                idxs.append(best_j)
            return idxs
        # Build AB pool
        try:
            rc_idx = headers.index("route_class")
        except ValueError:
            rc_idx = -1
        ab_rows = [r for r in kept if rc_idx >= 0 and r[rc_idx] in ("A", "B")]
        if len(ab_rows) >= 40:
            sel = _greedy_maxmin([r[0] for r in ab_rows], 40)
            chosen_div = [ab_rows[i] for i in sel]
        else:
            chosen_div = list(ab_rows)
            # top up from C by design_score
            c_rows = [r for r in kept if rc_idx >= 0 and r[rc_idx] == "C"]
            c_rows.sort(key=lambda x: float(x[36]) if x[36] is not None else 0.0, reverse=True)
            for r in c_rows:
                if len(chosen_div) >= 40:
                    break
                chosen_div.append(r)
        # Write diverse40 CSV/SMI
        div_csv = os.path.join(paths["outputs"], f"{pathogen}_amide_library_diverse40.csv")
        div_smi = os.path.join(paths["outputs"], f"{pathogen}_amide_library_diverse40.smi")
        div_sdf = os.path.join(paths["outputs"], f"{pathogen}_amide_library_diverse40.sdf")
        with open(div_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in chosen_div:
                w.writerow(r)
        with open(div_smi, "w", encoding="utf-8") as f:
            for r in chosen_div:
                f.write(r[0] + "\n")
        # Write SDF with amide CN indices as mol property
        try:
            if Chem is not None:
                from rdkit.Chem import SDWriter
                w = SDWriter(div_sdf)
                cprod_idx = headers.index("coupling_atoms_product")
                for r in chosen_div:
                    smi = r[0]
                    m = Chem.MolFromSmiles(smi)
                    if not m:
                        continue
                    rec = r[cprod_idx]
                    # rec format: C_idx=..;N_idx=..
                    cn = {kv.split('=')[0].strip(): kv.split('=')[1].strip() for kv in rec.split(';') if '=' in kv}
                    cn_str = f"C_idx={cn.get('C_idx','')};N_idx={cn.get('N_idx','')}"
                    m.SetProp("amide_CN_idx", cn_str)
                    w.write(m)
                w.close()
        except Exception:
            pass
        # Report (Top-20)
        if write_report:
            out_md = os.path.join(paths["reports"], f"{pathogen}_summary_report.md")
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(f"# {pathogen} Amide Generation Summary\n\n")
                f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")
                f.write(f"Inputs: safe_library={lib_path}\n\n")
                if median_div is not None:
                    f.write(f"Median ECFP4 Tanimoto (kept): {median_div:.3f}\n\n")
                    if median_div < 0.25:
                        f.write("WARNING: Low structural diversity; many products share the same acyl donor.\n\n")
                # Latent telemetry
                if len(kept) > 0:
                    pct_latent = 100.0 * (latent_used_count / len(kept))
                    f.write(f"Products using latent acids: {latent_used_count}/{len(kept)} ({pct_latent:.1f}%)\n\n")
                    if leaving_groups_counts:
                        f.write("Top latent leaving groups (name: count):\n\n")
                        top5 = sorted(leaving_groups_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
                        for name, cnt in top5:
                            f.write(f"- {name}: {cnt}\n")
                        f.write("\n")
                # Origin counts
                try:
                    rc_idx = headers.index("route_class")
                    hoa_idx = headers.index("handle_origin_acid")
                    src_idx = headers.index("acid_source_csv")
                except ValueError:
                    rc_idx = hoa_idx = src_idx = -1
                if rc_idx >= 0 and hoa_idx >= 0:
                    native_c = sum(1 for r in kept if r[hoa_idx] == "native_acid")
                    carbox_c = sum(1 for r in kept if r[hoa_idx] == "carboxylate")
                    easy_b = sum(1 for r in kept if r[rc_idx] == "B")
                    hard_c = sum(1 for r in kept if r[rc_idx] == "C")
                    ext_c = sum(1 for r in kept if src_idx >= 0 and r[src_idx] == "external_panel")
                    f.write("Counts by origin (kept):\n\n")
                    f.write(f"- native: {native_c}\n- carboxylate: {carbox_c}\n- easy_latent: {easy_b}\n- hard_latent: {hard_c}\n")
                    if ext_c:
                        f.write(f"- external_panel: {ext_c}\n")
                    f.write("\n")
                # Data limitation note
                if (acid_ratio < 0.23) or external_acid_products > 0:
                    f.write("Note: Acid scarcity detected in XAI-positive pool; latent esters and/or external acid panel used to reach target.\n\n")
                # EC: diverse Top-20 via greedy max-min
                def _greedy_maxmin(smiles_list: List[str], k: int = 20) -> List[int]:
                    if not Chem or not smiles_list:
                        return list(range(min(k, len(smiles_list))))
                    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles_list]
                    idxs: List[int] = []
                    if not fps:
                        return idxs
                    idxs.append(0)
                    while len(idxs) < min(k, len(fps)):
                        best_j = None
                        best_min_sim = 2.0
                        for j in range(len(fps)):
                            if j in idxs:
                                continue
                            sims = [DataStructs.TanimotoSimilarity(fps[j], fps[i]) for i in idxs]
                            min_sim = min(sims) if sims else 0.0
                            if min_sim < best_min_sim:
                                best_min_sim = min_sim
                                best_j = j
                        if best_j is None:
                            break
                        idxs.append(best_j)
                    return idxs
                # Build column index map for reporting
                idx = {k: headers.index(k) for k in [
                    "design_score","MW","LogP","TPSA","QED","SA_score",
                    "acid_role","amine_role","handle_origin_acid","acid_state",
                    "handle_origin_amine","amine_state","novel_vs_pathogen_train","novel_vs_union_train"
                ]}
                if pathogen == "EC":
                    sel = _greedy_maxmin([r[0] for r in kept], k=20)
                    chosen = [kept[i] for i in sel]
                    med20 = _median_tanimoto([r[0] for r in chosen])
                    if med20 is not None:
                        f.write(f"Diverse Top-20 median ECFP4 Tanimoto: {med20:.3f}\n\n")
                    f.write("Top-20 (diverse, greedy max-min)\n\n")
                    f.write("product_smiles | design_score | MW | LogP | TPSA | QED | SA_score | acid_role | amine_role | handle_acid(origin/state) | handle_amine(origin/state) | novel_P | novel_U | rationale\n")
                    f.write("---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---\n")
                    for r in chosen:
                        rationale = "scaffold acid; substituent amine; accessibility heuristic"
                        f.write(
                            f"{r[0]} | {r[idx['design_score']]} | {r[idx['MW']]} | {r[idx['LogP']]} | {r[idx['TPSA']]} | {r[idx['QED']]} | {r[idx['SA_score']]} | {r[idx['acid_role']]} | {r[idx['amine_role']]} | {r[idx['handle_origin_acid']]}/{r[idx['acid_state']]} | {r[idx['handle_origin_amine']]}/{r[idx['amine_state']]} | {r[idx['novel_vs_pathogen_train']]} | {r[idx['novel_vs_union_train']]} | {rationale}\n"
                        )
                else:
                    f.write("Top-20 (by design_score)\n\n")
                    f.write("product_smiles | design_score | MW | LogP | TPSA | QED | SA_score | acid_role | amine_role | handle_acid(origin/state) | handle_amine(origin/state) | novel_P | novel_U | rationale\n")
                    f.write("---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---\n")
                    for r in kept[:20]:
                        rationale = "scaffold acid; substituent amine; accessibility heuristic"
                        f.write(
                            f"{r[0]} | {r[idx['design_score']]} | {r[idx['MW']]} | {r[idx['LogP']]} | {r[idx['TPSA']]} | {r[idx['QED']]} | {r[idx['SA_score']]} | {r[idx['acid_role']]} | {r[idx['amine_role']]} | {r[idx['handle_origin_acid']]}/{r[idx['acid_state']]} | {r[idx['handle_origin_amine']]}/{r[idx['amine_state']]} | {r[idx['novel_vs_pathogen_train']]} | {r[idx['novel_vs_union_train']]} | {rationale}\n"
                        )

            # Diverse40 report
            div_md = os.path.join(paths["reports"], f"{pathogen}_diverse40_report.md")
            with open(div_md, "w", encoding="utf-8") as f:
                f.write(f"# {pathogen} Diverse40 Summary\n\n")
                f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")
                med40 = _median_tanimoto([r[0] for r in chosen_div]) if chosen_div else None
                if med40 is not None:
                    f.write(f"Median ECFP4 Tanimoto (diverse40): {med40:.3f}\n\n")
                f.write(f"Size: {len(chosen_div)}\n\n")
                # Medians for MW, LogP, TPSA, QED, SA_score and route composition of diverse40
                try:
                    import numpy as np
                    mw_idx = headers.index("MW"); lp_idx = headers.index("LogP"); tpsa_idx = headers.index("TPSA")
                    qed_idx = headers.index("QED"); sa_idx = headers.index("SA_score"); rc_idx = headers.index("route_class")
                    def med(vals):
                        vv=[float(v) for v in vals if v not in (None,'','None')]
                        return (np.median(vv) if vv else None)
                    mw_med = med([r[mw_idx] for r in chosen_div])
                    lp_med = med([r[lp_idx] for r in chosen_div])
                    tpsa_med = med([r[tpsa_idx] for r in chosen_div])
                    qed_med = med([r[qed_idx] for r in chosen_div])
                    sa_med = med([r[sa_idx] for r in chosen_div])
                    f.write("Medians (diverse40):\n\n")
                    f.write(f"- MW: {mw_med:.3f if mw_med is not None else 'NA'}\n")
                    f.write(f"- LogP: {lp_med:.3f if lp_med is not None else 'NA'}\n")
                    f.write(f"- TPSA: {tpsa_med:.3f if tpsa_med is not None else 'NA'}\n")
                    f.write(f"- QED: {qed_med:.3f if qed_med is not None else 'NA'}\n")
                    f.write(f"- SA_score: {sa_med:.3f if sa_med is not None else 'NA'}\n\n")
                    a_cnt = sum(1 for r in chosen_div if r[rc_idx]=='A')
                    b_cnt = sum(1 for r in chosen_div if r[rc_idx]=='B')
                    c_cnt = sum(1 for r in chosen_div if r[rc_idx]=='C')
                    f.write(f"Route composition (diverse40): A={a_cnt}, B={b_cnt}, C={c_cnt}\n")
                except Exception:
                    pass

        # Logs continuation
        logw("Generation:")
        logw(f"  Reaction: {REACTION_SMIRKS}")
        logw(f"  Attempts: {attempts} | successes: {successes} | site_forcing_failed: {site_failures} | valency_fixed: {valency_fixes} | cleaned: {cleanings}")
        logw("Safety & novelty:")
        if prejoin_safety_skipped:
            logw(f"  Pre-join safety skipped pairs (TRIPLE): {prejoin_safety_skipped}")
        if prejoin_latent_disallowed:
            logw(f"  Pre-join latent disallowed pairs: {prejoin_latent_disallowed}")
            if prejoin_latent_motif_counts:
                for k in sorted(prejoin_latent_motif_counts.keys()):
                    logw(f"    latent_disallowed motif {k}: {prejoin_latent_motif_counts[k]}")
        if prejoin_predicted_michael_acceptor:
            logw(f"  Pre-join predicted Michael acceptor (EC): {prejoin_predicted_michael_acceptor}")
        logw(f"  Excluded by safety: {safety_excluded}")
        if safety_alert_counts:
            logw("  Safety alert counts:")
            for k in sorted(safety_alert_counts.keys()):
                logw(f"    {k}: {safety_alert_counts[k]}")
        logw(f"  Excluded by novelty: {novelty_excluded} (pathogen_set={novelty_excluded_pathogen}, union_set={novelty_excluded_union})")
        logw(f"  Kept after novelty: {len(kept)} | removed duplicates: {removed_duplicates}")
        logw("Handle site counts:")
        logw(f"  native_acid_sites_considered: {c_native}")
        logw(f"  carboxylate_sites_considered: {c_carbox}")
        logw(f"  latent_ester_sites_considered: {c_latent}")
        logw(f"  neutral_amine_sites_considered: {n_neutral}")
        logw(f"  protonated_amine_sites_considered: {n_prot}")
        if median_div is not None:
            logw(f"Diversity: median_ECFP4_Tanimoto={median_div:.3f}")
        if side_frag_total:
            logw(f"Side fragments removed (total): {side_frag_total}")
            if side_frag_examples:
                logw(f"  Examples: {', '.join(side_frag_examples[:10])}")
        if route_reason_counts:
            logw("Route B/C reason breakdown:")
            for k in sorted(route_reason_counts.keys()):
                logw(f"  {k}: {route_reason_counts[k]}")
        # External panel contribution breakdown
        if external_acid_products:
            logw(f"External acid panel contribution: {external_acid_products}/{len(kept)} ({external_acid_products/len(kept)*100:.1f}%)")
            # acid_products holds per-acid counts; list external IDs
            ext_list = [aid for aid in external_acid_ids]
            for aid in ext_list:
                cnt = acid_products.get(aid, 0)
                if cnt:
                    logw(f"  external acid {aid}: {cnt}")
        logw("Outputs:")
        logw(f"  .smi: {out_smi}")
        logw(f"  .csv: {out_csv}")
        logw(f"  failures.csv: {out_fail}")
        if write_report:
            logw(f"  .md: {out_md}")
        # Backfill QED/SA for existing detailed CSVs in outputs (including current)
        try:
            filled_qed = filled_sa = total_rows = 0
            import glob
            for path in glob.glob(os.path.join(paths["outputs"], "*_amide_library_detailed.csv")):
                try:
                    rows = []
                    with open(path, "r", encoding="utf-8", newline="") as f:
                        rdr = csv.reader(f)
                        header = next(rdr)
                        # Expect QED and SA_score columns at indexes -3 and -2 respectively
                        for row in rdr:
                            total_rows += 1
                            try:
                                qi = header.index("QED")
                                si = header.index("SA_score")
                            except Exception:
                                rows.append(row)
                                continue
                            qv = row[qi]
                            sv = row[si]
                            if (qv == "" or qv == "None") or (sv == "" or sv == "None"):
                                qsmi = row[0]
                                q, s = _compute_qed_sa(qsmi)
                                if (qv == "" or qv == "None") and q is not None:
                                    row[qi] = str(q)
                                    filled_qed += 1
                                if (sv == "" or sv == "None") and s is not None:
                                    row[si] = str(s)
                                    filled_sa += 1
                            rows.append(row)
                    # rewrite file only if changed
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(header)
                        for r in rows:
                            w.writerow(r)
                except Exception:
                    continue
            # Also compute how many had valid QED/SA during this run
            qed_ok = sum(1 for r in kept if r[38] not in (None, "", "None"))
            sa_ok = sum(1 for r in kept if r[39] not in (None, "", "None"))
            logw(f"QED computed: {qed_ok} / {len(kept)} | SA_score computed: {sa_ok} / {len(kept)}")
            logw(f"QED backfilled: {filled_qed} / {total_rows} rows | SA_score backfilled: {filled_sa} / {total_rows} rows")
        except Exception:
            logw("QED/SA backfill encountered an error; continuing.")
        # Route composition targets reporting & donor spread
        try:
            rc_idx = headers.index("route_class")
            a_cnt = sum(1 for r in kept if r[rc_idx] == "A")
            b_cnt = sum(1 for r in kept if r[rc_idx] == "B")
            c_cnt = sum(1 for r in kept if r[rc_idx] == "C")
            if a_cnt < 20:
                logw(f"Target note: A target unmet: available A={a_cnt}")
            if b_cnt < 30 or b_cnt > 60:
                logw(f"Target note: B count={b_cnt} (target 30–60)")
            if c_cnt > 20:
                logw(f"Target note: C exceeds 20: count={c_cnt}")
            # Donor spread
            acid_id_idx = headers.index("acid_fragment_id")
            unique_donors = len(set(r[acid_id_idx] for r in kept))
            logw(f"Acid donor spread: unique_donors={unique_donors}")
            min_goal = 4
            try:
                # infer from CLI if available via local var (not passed here); keep 4 as soft default
                pass
            except Exception:
                pass
            if unique_donors < 4:
                logw("donor_spread_warning: fewer than 4 unique acid donors used")
        except Exception:
            pass
        logw("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amide-only molecule generator with per-pathogen safe libraries, novelty, and traceability",
        epilog="""
Library Types:
  SA, EC, CA           - Pathogen-specific (active against ONE pathogen only)
  SA_EC, SA_CA, CA_EC  - Dual-active (active against exactly TWO pathogens)
  TRIPLE               - Triple-active (active against ALL THREE pathogens)
        """
    )
    parser.add_argument(
        "--pathogen",
        required=True,
        choices=["SA", "EC", "CA", "SA_EC", "SA_CA", "CA_EC", "TRIPLE"],
        help="Library type identifier"
    )
    parser.add_argument("--fragments-root", required=False, help="Unused here; present for interface parity")
    parser.add_argument("--outdir", default="FBDD")
    parser.add_argument("--sa-train", default=os.path.join("FBDD", "S_aureus_input.csv"))
    parser.add_argument("--ec-train", default=os.path.join("FBDD", "E_coli_input.csv"))
    parser.add_argument("--ca-train", default=os.path.join("FBDD", "C_albicans_input.csv"))
    parser.add_argument("--max-acids", type=int, default=None)
    parser.add_argument("--max-amines", type=int, default=None)
    parser.add_argument("--minmax-props", type=str, default=None, help="Optional JSON window (unused stub for now)")
    parser.add_argument("--allow-latent-acids", action="store_true")
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--max-products-per-acid", type=int, default=25)
    parser.add_argument("--min-acid-donors", type=int, default=4, help="Soft goal for minimum unique acid donors; logs warning if unmet")
    parser.add_argument("--score-with-model", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_products(
        pathogen=args.pathogen,
        base_outdir=args.outdir,
        sa_train=args.sa_train,
        ec_train=args.ec_train,
        ca_train=args.ca_train,
        max_acids=args.max_acids,
        max_amines=args.max_amines,
        allow_latent_acids=args.allow_latent_acids,
        pair_batch_size=args.pair_batch_size,
        target_count=args.target_count,
        max_products_per_acid=args.max_products_per_acid,
        score_with_model=args.score_with_model,
        write_report=args.write_report,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
