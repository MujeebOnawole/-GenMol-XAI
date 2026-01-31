import pandas as pd

# Load data
input_df = pd.read_csv('data/genmol_input.csv')
ec_pred = pd.read_csv('data/predictions/genmol_all_input_EC_prediction.csv')
sa_pred = pd.read_csv('data/predictions/genmol_all_input_SA_prediction.csv')
ca_pred = pd.read_csv('data/predictions/genmol_all_input_CA_prediction.csv')
existing = pd.read_csv('data/predictions/discussed_compounds_v4_CORRECTED.csv')

def create_row(cmpd_id, target_pathogen):
    inp = input_df[input_df['COMPOUND_ID'] == cmpd_id].iloc[0]
    ec = ec_pred[ec_pred['COMPOUND_ID'] == cmpd_id].iloc[0]
    sa = sa_pred[sa_pred['COMPOUND_ID'] == cmpd_id].iloc[0]
    ca = ca_pred[ca_pred['COMPOUND_ID'] == cmpd_id].iloc[0]

    # Determine reaction conditions based on route
    if inp['route_class'] == 'B':
        conditions = 'K2CO3, MeOH/DMF, 60C, 6h'
        conditions_short = 'K2CO3, 60C'
        mechanism = 'Ester aminolysis (mild)'
        ester_detail = 'latent_ester (methyl/ethyl ester)'
        linkage = 'AMIDE (N-C(=O)-C)'
    else:  # Route C
        conditions = 'Neat amine, 80-120C, 12-24h'
        conditions_short = 'Neat, 100C'
        mechanism = 'Oxazolidinone ring-opening -> UREA formation'
        ester_detail = 'cyclic carbamate (oxazolidinone)'
        linkage = 'UREA (N-C(=O)-N)'

    return {
        'compound_id': cmpd_id,
        'target_pathogen': target_pathogen,
        'source_library': inp['source_library'],
        'product_smiles_mapped': inp['SMILES'],
        'product_smiles_clean': inp['SMILES'].replace('[N:5]', 'N').replace('[C:1]', 'C'),
        'product_inchikey': inp['product_inchikey'],
        'route_class': inp['route_class'],
        'reaction_type': inp['reaction_type'],
        'acid_fragment_id': inp['acid_fragment_id'],
        'acid_original_smiles': inp['acid_fragment_smiles'],
        'acid_source_file': f"{inp['source_library']}_positive_substituents.csv",
        'acid_with_handle_smiles': inp['acid_fragment_smiles'],
        'acid_handle_origin': inp['handle_origin_acid'],
        'acid_state': inp['acid_state'],
        'acid_role': inp['acid_role'],
        'acid_avg_attribution': inp['acid_avg_attribution'],
        'acid_activity_rate': inp['acid_activity_rate'],
        'amine_fragment_id': inp['amine_fragment_id'],
        'amine_original_smiles': inp['amine_fragment_smiles'],
        'amine_source_file': f"{inp['source_library']}_positive_substituents.csv",
        'amine_with_handle_smiles': inp['amine_fragment_smiles'],
        'amine_handle_origin': inp['handle_origin_amine'],
        'amine_state': inp['amine_state'],
        'amine_role': inp['amine_role'],
        'amine_avg_attribution': inp['amine_avg_attribution'],
        'amine_activity_rate': inp['amine_activity_rate'],
        'MW': inp['MW'],
        'LogP': inp['LogP'],
        'TPSA': inp['TPSA'],
        'HBA': inp['HBA'],
        'HBD': inp['HBD'],
        'AromRings': inp['AromRings'],
        'RotBonds': inp['RotBonds'],
        'QED': inp['QED'],
        'SA_score': inp['SA_score'],
        'SA_prediction': 1 if sa['ensemble_prediction'] >= 0.5 else 0,
        'SA_probability': sa['ensemble_prediction'],
        'SA_scenario': sa['decision_scenario'],
        'EC_prediction': 1 if ec['ensemble_prediction'] >= 0.5 else 0,
        'EC_probability': ec['ensemble_prediction'],
        'EC_scenario': ec['decision_scenario'],
        'CA_prediction': 1 if ca['ensemble_prediction'] >= 0.5 else 0,
        'CA_probability': ca['ensemble_prediction'],
        'CA_scenario': ca['decision_scenario'],
        'novel_vs_pathogen_train': inp['novel_vs_pathogen_train'],
        'novel_vs_union_train': inp['novel_vs_union_train'],
        'reaction_conditions': conditions,
        'reaction_conditions_short': conditions_short,
        'reaction_mechanism': mechanism,
        'ester_type_detail': ester_detail,
        'product_linkage': linkage,
        'acid_qmol_signature': '',
        'acid_qmol_node_class': 'ring',
        'acid_qmol_ring_count': 1.0,
        'acid_qmol_aromatic': False,
        'acid_qmol_heteroatoms': '',
        'amine_qmol_signature': '',
        'amine_qmol_node_class': 'ring',
        'amine_qmol_ring_count': 1.0,
        'amine_qmol_aromatic': True,
        'amine_qmol_heteroatoms': 'N',
        'SA_reliability_avg_attribution': sa['reliability_avg_attribution'],
        'SA_top_substructure_1': sa.get('murcko_substructure_0_smiles', ''),
        'SA_top_attribution_1': sa.get('murcko_substructure_0_attribution', 0),
        'SA_top_substructure_2': sa.get('murcko_substructure_1_smiles', ''),
        'SA_top_attribution_2': sa.get('murcko_substructure_1_attribution', 0),
        'SA_top_substructure_3': sa.get('murcko_substructure_2_smiles', ''),
        'SA_top_attribution_3': sa.get('murcko_substructure_2_attribution', 0),
        'EC_reliability_avg_attribution': ec['reliability_avg_attribution'],
        'EC_top_substructure_1': ec.get('murcko_substructure_0_smiles', ''),
        'EC_top_attribution_1': ec.get('murcko_substructure_0_attribution', 0),
        'EC_top_substructure_2': ec.get('murcko_substructure_1_smiles', ''),
        'EC_top_attribution_2': ec.get('murcko_substructure_1_attribution', 0),
        'EC_top_substructure_3': ec.get('murcko_substructure_2_smiles', ''),
        'EC_top_attribution_3': ec.get('murcko_substructure_2_attribution', 0),
        'CA_reliability_avg_attribution': ca['reliability_avg_attribution'],
        'CA_top_substructure_1': ca.get('murcko_substructure_0_smiles', ''),
        'CA_top_attribution_1': ca.get('murcko_substructure_0_attribution', 0),
        'CA_top_substructure_2': ca.get('murcko_substructure_1_smiles', ''),
        'CA_top_attribution_2': ca.get('murcko_substructure_1_attribution', 0),
        'CA_top_substructure_3': ca.get('murcko_substructure_2_smiles', ''),
        'CA_top_attribution_3': ca.get('murcko_substructure_2_attribution', 0),
        'acid_qmol_enrichment_context': f'{target_pathogen}-specific fragment',
        'amine_qmol_enrichment_context': f'{target_pathogen}-enriched scaffold'
    }

# Create rows for all three exemplars
sa_row = existing[existing['compound_id'] == 'CMPD_096'].iloc[0].to_dict()  # Keep original SA
ec_row = create_row('CMPD_162', 'EC')
ca_row = create_row('CMPD_428', 'CA')

# Create final dataframe
final = pd.DataFrame([sa_row, ec_row, ca_row])
final.to_csv('data/predictions/discussed_compounds_v5_CORRECTED.csv', index=False)

print('UPDATED EXEMPLAR COMPOUNDS')
print('='*80)
for idx, row in final.iterrows():
    print(f"{row['compound_id']}: {row['target_pathogen']} (from {row['source_library']} library)")
    print(f"   Predictions: SA={row['SA_probability']:.3f}, EC={row['EC_probability']:.3f}, CA={row['CA_probability']:.3f}")
    print()
