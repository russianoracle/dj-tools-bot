"""Check cached features."""

import pickle
import numpy as np

cache_file = "models/checkpoints/features.pkl"

print(f"Loading cache from: {cache_file}")

with open(cache_file, 'rb') as f:
    cached_data = pickle.load(f)

print(f"\nCache type: {type(cached_data)}")

import pandas as pd

if isinstance(cached_data, pd.DataFrame):
    print(f"\n✓ DataFrame with {len(cached_data)} rows")
    print(f"Columns: {list(cached_data.columns)}")

    # Check 'features_list' column
    if 'features_list' in cached_data.columns:
        features_list = cached_data['features_list']
        print(f"\nFeatures list column type: {type(features_list)}")

        # Check first feature
        if len(features_list) > 0:
            first_feature = features_list.iloc[0]
            print(f"First feature type: {type(first_feature)}")

            # Try to convert to vector
            if hasattr(first_feature, 'to_vector'):
                vec = first_feature.to_vector(include_embeddings=False)
                print(f"\n✓ Feature vector shape: {vec.shape}")
                print(f"✓ Number of features: {len(vec)}")

                if len(vec) == 10:
                    print("\n⚠️  OLD CACHE: 10 features (FastZoneFeatureExtractor)")
                elif len(vec) == 51:
                    print("\n✓ NEW CACHE: 51 features (ZoneFeatureExtractor with DJ features)")
                else:
                    print(f"\n? UNKNOWN: {len(vec)} features")

                print(f"\nFirst 15 feature values: {vec[:15]}")
            else:
                print(f"Feature object: {first_feature}")
    else:
        print("\n⚠️  No 'features_list' column found")

elif isinstance(cached_data, dict):
    print(f"Cache keys: {cached_data.keys()}")

    if 'features_list' in cached_data:
        features_list = cached_data['features_list']
        print(f"\nFeatures list type: {type(features_list)}")
        print(f"Number of tracks: {len(features_list)}")

        # Check first feature
        if len(features_list) > 0:
            first_feature = features_list.iloc[0] if hasattr(features_list, 'iloc') else features_list[0]
            print(f"\nFirst feature type: {type(first_feature)}")

            # Try to convert to vector
            if hasattr(first_feature, 'to_vector'):
                vec = first_feature.to_vector(include_embeddings=False)
                print(f"\n✓ Feature vector shape: {vec.shape}")
                print(f"✓ Number of features: {len(vec)}")

                if len(vec) == 10:
                    print("\n⚠️  OLD CACHE: 10 features (FastZoneFeatureExtractor)")
                elif len(vec) == 51:
                    print("\n✓ NEW CACHE: 51 features (ZoneFeatureExtractor with DJ features)")
                else:
                    print(f"\n? UNKNOWN: {len(vec)} features")

                print(f"\nFirst 15 feature values: {vec[:15]}")
            else:
                print(f"Feature object: {first_feature}")
