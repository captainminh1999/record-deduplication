import pandas as pd

# Load the similarity features and pairs data
features_df = pd.read_csv('data/outputs/features.csv')
pairs_df = pd.read_csv('data/outputs/pairs.csv')

# Record IDs to analyze
record_groups = [
    ["06a77fa687c76550852d0e170cbb35ad", "e500daec1b8f6850d5d38738dc4bcb85", "4e001eec1b8f6850d5d38738dc4bcbca"],
    ["82cafcc41b7380108321311d0d4bcbc7", "c81052201bcf6850d5d38738dc4bcb46", "3c4d22adc3a392d01a9c742a050131c7"],
    ["0f3b56b01b3665101c3ddc6fcc4bcbe4", "9f2ad3e9c39a9a901a9c742a0501313d", "55a044501b6f5d90b37020e6b04bcbe5", "9998f2061b3139d0d1f74119b04bcb7d", "46dd74981bf1f990d1f74119b04bcb1e"]
]

print("=== SIMILARITY PAIRS ANALYSIS ===\n")

for i, group in enumerate(record_groups, 1):
    print(f"Group {i}: Checking pairwise similarities")
    print("-" * 60)
    
    # Check all pairs within this group
    found_pairs = []
    for j, rec1 in enumerate(group):
        for k, rec2 in enumerate(group):
            if j < k:  # Avoid duplicates and self-pairs
                # Check if this pair exists in either direction
                pair1 = features_df[(features_df['record_id_1'] == rec1) & (features_df['record_id_2'] == rec2)]
                pair2 = features_df[(features_df['record_id_1'] == rec2) & (features_df['record_id_2'] == rec1)]
                
                if len(pair1) > 0:
                    found_pairs.append((rec1, rec2, pair1.iloc[0]))
                elif len(pair2) > 0:
                    found_pairs.append((rec2, rec1, pair2.iloc[0]))
    
    if found_pairs:
        print(f"  Found {len(found_pairs)} similarity pairs:")
        for rec1, rec2, pair_data in found_pairs:
            print(f"    {rec1[:8]}... ↔ {rec2[:8]}...")
            print(f"      Company sim: {pair_data['company_sim']:.3f}")
            print(f"      Domain sim:  {pair_data['domain_sim']:.3f}")
            print(f"      Phone exact: {pair_data['phone_exact']:.3f}")
            print(f"      Address sim: {pair_data['address_sim']:.3f}")
            print()
    else:
        print("  ❌ NO SIMILARITY PAIRS FOUND between these records!")
        print("     This explains why they're in different clusters.")
        print("     The blocking step may have filtered them out.")
    
    print("\n" + "="*80 + "\n")

# Also check if any of these records appear in pairs at all
print("=== CHECKING IF RECORDS APPEAR IN ANY PAIRS ===\n")
all_records = [rec for group in record_groups for rec in group]

for record_id in all_records:
    pairs_as_1 = features_df[features_df['record_id_1'] == record_id]
    pairs_as_2 = features_df[features_df['record_id_2'] == record_id]
    total_pairs = len(pairs_as_1) + len(pairs_as_2)
    
    print(f"Record {record_id[:12]}... appears in {total_pairs} pairs")
    if total_pairs == 0:
        print("  ⚠️  This record has NO similarity pairs at all!")
    elif total_pairs < 5:
        print("  ⚠️  This record has very few similarity pairs")
    print()
