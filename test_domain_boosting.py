#!/usr/bin/env python3
"""
Test the fixed domain boosting to ensure unique values are preserved
"""
import numpy as np
import pandas as pd

def test_domain_boosting():
    print("🧪 TESTING FIXED DOMAIN BOOSTING")
    print("================================")
    
    # Create a test dataset that simulates perfect domain matches
    test_data = pd.DataFrame({
        'record_id_1': ['A', 'A', 'B', 'B', 'C'],
        'record_id_2': ['B', 'C', 'C', 'D', 'D'], 
        'domain_sim': [1.0, 1.0, 1.0, 1.0, 1.0],  # All perfect domain matches
        'company_sim': [0.8, 0.6, 0.9, 0.7, 0.5]
    })
    
    print("📊 Original test data:")
    print(test_data)
    
    # Simulate the adaptive clusterer's domain boosting logic
    melted = test_data.copy()
    
    # Apply the fixed boosting logic
    if "domain_sim" in melted.columns:
        melted["domain_sim"] = melted["domain_sim"] * 1000.0  # Ultra-high weight for domain
        print(f"\n📈 After 1000x boost:")
        print(melted[['domain_sim']])
        
        # Special boost for perfect domain matches, but preserve uniqueness for subdivision
        perfect_domain_mask = melted["domain_sim"] >= 1000.0  # After weighting, perfect match = 1000.0
        
        if perfect_domain_mask.sum() > 0:
            # Add small incremental values (0.001, 0.002, etc.) to preserve uniqueness
            unique_offsets = np.arange(perfect_domain_mask.sum()) * 0.001
            melted.loc[perfect_domain_mask, "domain_sim"] = 4999.0 + unique_offsets
            print(f"\n🎯 After unique boost preservation:")
            print(melted[['domain_sim']])
            print(f"   Range: {melted['domain_sim'].min():.6f} - {melted['domain_sim'].max():.6f}")
            print(f"   Unique values: {melted['domain_sim'].nunique()}")
            
            # Test subdivision-friendly detection
            max_val = melted['domain_sim'].max()
            min_val = melted['domain_sim'].min()
            
            print(f"\n🔍 Subdivision Detection Test:")
            print(f"   Max value: {max_val}")
            print(f"   Min value: {min_val}")
            
            # Case 1: Uniform high values (old problematic case)
            if max_val == min_val and max_val >= 4999.0:
                print("   ❌ UNIFORM BOOSTED VALUES - Would be preserved (no subdivision)")
            # Case 2: High domain values with variation (new fixed case)  
            elif max_val >= 4999.0 and min_val >= 4999.0:
                print("   ✅ BOOSTED VALUES WITH VARIATION - Subdivision allowed!")
            else:
                print("   ℹ️  Non-boosted values")
                
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_domain_boosting()
