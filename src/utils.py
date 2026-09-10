import yaml
from PIL import Image
import selfies as sf
import deepsmiles
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# Mol rep utility functions
# ============================================================================

def _parse_to_mol(rep: str, rep_type: str) -> Chem.Mol | None:
    """
    Parse a molecular representation string to an RDKit Mol object.

    Args:
        rep: Molecular representation string
        rep_type: Representation type (can_smiles, can_selfies, can_deepsmiles, inchi)

    Returns:
        RDKit Mol object, or None if parsing fails
    """
    if not rep or not isinstance(rep, str):
        return None

    rep = rep.strip()
    
    try:
        match rep_type:
            case "can_smiles":
                return Chem.MolFromSmiles(rep)
            case "can_selfies":
                decoded_smiles = sf.decoder(rep)
                return Chem.MolFromSmiles(decoded_smiles)
            case "can_deepsmiles":
                converter = deepsmiles.Converter(rings=True, branches=True)
                decoded_smiles = converter.decode(rep)
                return Chem.MolFromSmiles(decoded_smiles)
            case "inchi":
                return Chem.MolFromInchi(rep)
            case _:
                raise ValueError(f"Unsupported rep_type: {rep_type}")
    except Exception:
        return None


def _get_canonical_smiles(mol: Chem.Mol) -> str | None:
    """Get canonical SMILES for a molecule."""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _get_inchikey(mol: Chem.Mol) -> str | None:
    """Get InChIKey for a molecule."""
    try:
        inchi = Chem.MolToInchi(mol)
        if inchi:
            # InChIKey 是 InChI 的哈希，格式为 14-10-1
            return Chem.InchiToInchiKey(inchi)
        return None
    except Exception:
        return None


def _compute_tanimoto(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute Tanimoto similarity between two molecules (Morgan fingerprint, radius=2, nBits=2048).
    """
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


# ============================================================================
# Public API: Molecular evaluation metrics
# ============================================================================

def compute_mol_metrics(gt_rep: str, pred_rep: str, rep_type: str) -> dict:
    """
    Compute molecular representation evaluation metrics.

    Args:
        gt_rep: Ground truth molecular representation
        pred_rep: Model-predicted molecular representation
        rep_type: Representation type (can_smiles, can_selfies, can_deepsmiles, inchi)

    Returns:
        Dictionary containing all metrics:
        - is_gt_valid: Whether GT is parsable
        - is_pred_valid: Whether Pred is parsable
        - is_em: Exact Match (string-level match)
        - is_can_smiles_match: Canonical SMILES match
        - is_inchikey_match: InChIKey match
        - tanimoto_sim: Tanimoto similarity (0.0-1.0)
        - _parse_status: Parsing status ("success", "gt_failed", "pred_failed", "both_failed", "missing")
    """
    # Handle missing inputs
    if gt_rep is None or pred_rep is None:
        return {
            "is_gt_valid": False,
            "is_pred_valid": False,
            "is_em": False,
            "is_can_smiles_match": False,
            "is_inchikey_match": False,
            "tanimoto_sim": 0.0,
            "_parse_status": "missing"
        }
    
    # 1. Exact Match (string comparison, parsing-independent)
    is_em = gt_rep.strip() == pred_rep.strip()
    
    # 2. Parse molecules
    gt_mol = _parse_to_mol(gt_rep, rep_type)
    pred_mol = _parse_to_mol(pred_rep, rep_type)
    
    is_gt_valid = gt_mol is not None
    is_pred_valid = pred_mol is not None
    
    # 3. Determine parse status
    if is_gt_valid and is_pred_valid:
        parse_status = "success"
    elif is_gt_valid:
        parse_status = "pred_failed"
    elif is_pred_valid:
        parse_status = "gt_failed"
    else:
        parse_status = "both_failed"
    
    # 4. Compute structural match metrics
    if is_gt_valid and is_pred_valid:
        gt_smiles = _get_canonical_smiles(gt_mol)
        pred_smiles = _get_canonical_smiles(pred_mol)
        is_can_smiles_match = (gt_smiles is not None and pred_smiles is not None and 
                               gt_smiles == pred_smiles)
        
        gt_inchikey = _get_inchikey(gt_mol)
        pred_inchikey = _get_inchikey(pred_mol)
        is_inchikey_match = (gt_inchikey is not None and pred_inchikey is not None and 
                             gt_inchikey == pred_inchikey)
        
        tanimoto_sim = _compute_tanimoto(gt_mol, pred_mol)
    else:
        # Parse failure: structural metrics return False/0.0, counted in denominator to avoid inflation
        is_can_smiles_match = False
        is_inchikey_match = False
        tanimoto_sim = 0.0
    
    return {
        "is_gt_valid": is_gt_valid,
        "is_pred_valid": is_pred_valid,
        "is_em": is_em,
        "is_can_smiles_match": is_can_smiles_match,
        "is_inchikey_match": is_inchikey_match,
        "tanimoto_sim": tanimoto_sim,
        "_parse_status": parse_status
    }