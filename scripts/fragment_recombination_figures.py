"""
Fragment Recombination Visualization Script
Creates publication-quality figures showing fragment recombination for exemplar compounds.
Color-codes fragments in both the reactants and products to show origin.
"""

import os
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import io

# Define exemplar compound data
EXEMPLARS = {
    'CMPD_087': {
        'compound_id': 'CMPD_087',
        'target_pathogen': 'S. aureus',
        'acid_smiles': 'CC(=O)NC[C@H]1CN(c2ccc(-c3ccc(C=N)c(O)c3)c(F)c2)C(=O)O1',
        'amine_smiles': 'CN[C@H]1CO[C@H](CCc2ccnc3ccc(OC)cc23)OC1',
        'product_smiles': 'COc1ccc2nccc(CC[C@H]3OC[C@H](N(C)C(=O)N(C[CH]CNC(C)=O)c4ccc(-c5ccc(C=N)c(O)c5)c(F)c4)CO3)c2c1',
        'route': 'C',
        'conditions': 'Neat, 100°C',
        'mechanism': 'Oxazolidinone ring-opening',
        'product_linkage': 'UREA',
        'acid_type': 'Oxazolidinone',
        'amine_type': '2° amine',
        'SA_prob': 0.982,
        'EC_prob': 0.361,
        'CA_prob': 0.199,
        'tier': 'Tier 3 (Scenario A)'
    },
    'CMPD_162': {
        'compound_id': 'CMPD_162',
        'target_pathogen': 'E. coli',
        'acid_smiles': 'C[C@@](C[C@H]1CNC(=O)O1)(C(=O)NO)S(C)(=O)=O',
        'amine_smiles': 'Cc1ccc2[nH]c(NCCC#N)nc2n1',
        'product_smiles': 'Cc1ccc2[nH]c(N(CCC#N)C(=O)NC[CH]C[C@](C)(C(=O)NO)S(C)(=O)=O)nc2n1',
        'route': 'C',
        'conditions': 'Neat, 100°C',
        'mechanism': 'Oxazolidinone ring-opening',
        'product_linkage': 'UREA',
        'acid_type': 'Oxazolidinone',
        'amine_type': '2° amine',
        'SA_prob': 0.271,
        'EC_prob': 0.987,
        'CA_prob': 0.250,
        'tier': 'Tier 3 (Scenario A)'
    },
    'CMPD_428': {
        'compound_id': 'CMPD_428',
        'target_pathogen': 'C. albicans',
        'acid_smiles': 'CCC(=O)OC',
        'amine_smiles': 'c1ccc(Nc2ccncc2)cc1',  # Neutralized pyridinium
        'product_smiles': 'CCC(=O)N(c1ccncc1)c1ccccc1',  # Neutralized
        'route': 'B',
        'conditions': 'K₂CO₃, 60°C',
        'mechanism': 'Ester aminolysis',
        'product_linkage': 'AMIDE',
        'acid_type': 'Methyl ester',
        'amine_type': '2° amine',
        'SA_prob': 0.013,
        'EC_prob': 0.044,
        'CA_prob': 0.776,
        'tier': 'Tier 3 (Scenario A)'
    }
}

# Color definitions (RGB tuples, normalized to 0-1 for RDKit)
ACID_COLOR = (0.0, 0.75, 1.0, 1.0)       # Deep sky blue / cyan
AMINE_COLOR = (0.6, 0.8, 0.2, 1.0)       # Yellow-green
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_molecule_with_color(mol, color, size=(500, 400)):
    """Draw molecule with all atoms highlighted in one color."""
    if mol is None:
        return None

    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True

    # Highlight all atoms with the same color
    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_colors = {i: color for i in highlight_atoms}
    highlight_radii = {i: 0.4 for i in highlight_atoms}

    # Highlight all bonds
    highlight_bonds = list(range(mol.GetNumBonds()))
    bond_colors = {i: color for i in highlight_bonds}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
        highlightAtomRadii=highlight_radii
    )

    drawer.FinishDrawing()

    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))
    return img


def draw_product_with_fragment_colors(product_mol, amine_mol, size=(600, 450)):
    """Draw product with atoms colored based on fragment origin."""
    if product_mol is None:
        return None

    rdDepictor.Compute2DCoords(product_mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True

    # Find amine fragment atoms in product
    amine_match = product_mol.GetSubstructMatch(amine_mol) if amine_mol else ()

    # Assign colors to atoms
    highlight_atoms = list(range(product_mol.GetNumAtoms()))
    highlight_colors = {}

    for i in highlight_atoms:
        if i in amine_match:
            highlight_colors[i] = AMINE_COLOR
        else:
            highlight_colors[i] = ACID_COLOR

    highlight_radii = {i: 0.4 for i in highlight_atoms}

    # Color bonds based on atom colors
    highlight_bonds = list(range(product_mol.GetNumBonds()))
    bond_colors = {}

    for bond in product_mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # If both atoms have same color, bond gets that color
        # Otherwise, bond is gray (interface between fragments)
        if highlight_colors.get(begin_idx) == highlight_colors.get(end_idx):
            bond_colors[bond_idx] = highlight_colors.get(begin_idx, ACID_COLOR)
        else:
            bond_colors[bond_idx] = (0.5, 0.5, 0.5, 1.0)  # Gray

    drawer.DrawMolecule(
        product_mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
        highlightAtomRadii=highlight_radii
    )

    drawer.FinishDrawing()

    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))
    return img


def create_recombination_figure(exemplar_data, output_path, figsize=(1800, 700)):
    """Create a single figure showing fragment recombination."""

    # Parse molecules
    acid_mol = Chem.MolFromSmiles(exemplar_data['acid_smiles'])
    amine_mol = Chem.MolFromSmiles(exemplar_data['amine_smiles'])
    product_mol = Chem.MolFromSmiles(exemplar_data['product_smiles'])

    if acid_mol is None:
        print(f"Error parsing acid SMILES: {exemplar_data['acid_smiles']}")
        return
    if amine_mol is None:
        print(f"Error parsing amine SMILES: {exemplar_data['amine_smiles']}")
        return
    if product_mol is None:
        print(f"Error parsing product SMILES: {exemplar_data['product_smiles']}")
        return

    # Create molecule images
    mol_size = (450, 380)
    product_size = (600, 450)

    # Draw acid fragment in cyan
    acid_img = draw_molecule_with_color(acid_mol, ACID_COLOR, mol_size)

    # Draw amine fragment in yellow-green
    amine_img = draw_molecule_with_color(amine_mol, AMINE_COLOR, mol_size)

    # Draw product with dual coloring
    product_img = draw_product_with_fragment_colors(product_mol, amine_mol, product_size)

    # Create the composite figure
    fig_width, fig_height = figsize
    fig = Image.new('RGB', (fig_width, fig_height), WHITE)
    draw = ImageDraw.Draw(fig)

    # Try to load fonts
    try:
        title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
        label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except:
        try:
            title_font = ImageFont.truetype("arial.ttf", 26)
            label_font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    # Layout parameters
    y_title = 15
    y_subtitle = 50
    y_mol = 100
    y_label = 500

    # Paste acid fragment
    x_acid = 30
    if acid_img:
        fig.paste(acid_img, (x_acid, y_mol))

    # Draw plus sign
    x_plus = x_acid + mol_size[0] + 10
    draw.text((x_plus, y_mol + mol_size[1]//2 - 20), "+", fill=BLACK, font=title_font)

    # Paste amine fragment
    x_amine = x_plus + 40
    if amine_img:
        fig.paste(amine_img, (x_amine, y_mol))

    # Draw arrow with conditions
    x_arrow_start = x_amine + mol_size[0] + 30
    x_arrow_end = x_arrow_start + 140
    y_arrow = y_mol + mol_size[1]//2

    # Draw arrow line
    draw.line([(x_arrow_start, y_arrow), (x_arrow_end - 15, y_arrow)], fill=BLACK, width=3)
    # Draw arrowhead
    draw.polygon([(x_arrow_end, y_arrow),
                  (x_arrow_end - 20, y_arrow - 10),
                  (x_arrow_end - 20, y_arrow + 10)], fill=BLACK)

    # Add conditions above arrow
    conditions_text = exemplar_data['conditions']
    route_text = f"Route {exemplar_data['route']}"
    draw.text((x_arrow_start + 15, y_arrow - 55), conditions_text, fill=BLACK, font=label_font)
    draw.text((x_arrow_start + 35, y_arrow - 30), route_text, fill=BLACK, font=label_font)

    # Paste product
    x_product = x_arrow_end + 20
    if product_img:
        fig.paste(product_img, (x_product, y_mol - 30))

    # Add labels below molecules
    # Acid label - cyan color
    acid_label1 = "Acid Fragment"
    acid_label2 = f"({exemplar_data['acid_type']})"
    acid_center = x_acid + mol_size[0]//2
    draw.text((acid_center - 70, y_label), acid_label1, fill=(0, 140, 200), font=label_font)
    draw.text((acid_center - 60, y_label + 25), acid_label2, fill=(0, 140, 200), font=small_font)

    # Amine label - green color
    amine_label1 = "Amine Fragment"
    amine_label2 = f"({exemplar_data['amine_type']})"
    amine_center = x_amine + mol_size[0]//2
    draw.text((amine_center - 70, y_label), amine_label1, fill=(100, 150, 30), font=label_font)
    draw.text((amine_center - 50, y_label + 25), amine_label2, fill=(100, 150, 30), font=small_font)

    # Product label with predictions
    target = exemplar_data['target_pathogen']
    if 'aureus' in target:
        pred_text = f"Pred SA: {exemplar_data['SA_prob']:.3f}"
    elif 'coli' in target:
        pred_text = f"Pred EC: {exemplar_data['EC_prob']:.3f}"
    else:
        pred_text = f"Pred CA: {exemplar_data['CA_prob']:.3f}"

    product_center = x_product + product_size[0]//2
    draw.text((product_center - 60, y_label + 30), pred_text, fill=BLACK, font=label_font)
    draw.text((product_center - 45, y_label + 55), exemplar_data['compound_id'], fill=BLACK, font=label_font)

    # Add title
    title = f"Figure: Computational fragment recombination of {exemplar_data['compound_id']} ({exemplar_data['target_pathogen']}-selective exemplar)"
    draw.text((fig_width//2 - 450, y_title), title, fill=BLACK, font=title_font)

    # Add prediction details
    pred_details = f"Predicted activity: SA={exemplar_data['SA_prob']:.3f}, EC={exemplar_data['EC_prob']:.3f}, CA={exemplar_data['CA_prob']:.3f} ({exemplar_data['tier']})"
    draw.text((fig_width//2 - 320, y_subtitle), pred_details, fill=BLACK, font=small_font)

    # Add linkage type label
    linkage_text = f"Product: {exemplar_data['product_linkage']} bond"
    draw.text((x_product + 30, y_mol - 25), linkage_text, fill=(100, 100, 100), font=small_font)

    # Save figure
    fig.save(output_path, dpi=(300, 300))
    print(f"Saved: {output_path}")

    return fig


def create_all_figures(output_dir):
    """Create recombination figures for all exemplar compounds."""
    os.makedirs(output_dir, exist_ok=True)

    for compound_id, data in EXEMPLARS.items():
        output_path = os.path.join(output_dir, f"{compound_id}_recombination.png")
        try:
            create_recombination_figure(data, output_path)
        except Exception as e:
            print(f"Error creating figure for {compound_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'results', 'recombination_figures')

    print("Creating fragment recombination figures...")
    create_all_figures(output_dir)
    print("\nDone! Figures saved to:", output_dir)
