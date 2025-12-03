import os
from src.loader import load_all_data


# from src.features import normalize_signature, extract_global_features
# from src.dtw_algo import calculate_dtw_distance
# from src.models import SignatureVerifier


def main():
    print("=== Signature Verification System (MCYT) ===")

    DATA_PATH = "data"

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' not found.")
        return

    print("Loading full dataset...")
    dataset = load_all_data(DATA_PATH)

    print("\n=== Dataset Summary ===")
    print("Total writers loaded:", len(dataset))

    first_writer = list(dataset.keys())[0]
    print("Example writer:", first_writer)
    print("Enrollment signatures:", len(dataset[first_writer]["enrollment"]))
    print("Verification signatures:", len(dataset[first_writer]["verification"]))

    # OPTIONAL: Demonstration of how C/D/B will use your data structure
    # writer_data = dataset[first_writer]
    # ref_sig = writer_data["enrollment"][0]
    # test_sig = writer_data["verification"][0]["df"]
    #
    # # B: preprocess
    # ref_sig = normalize_signature(ref_sig)
    # test_sig = normalize_signature(test_sig)
    #
    # # C: DTW
    # dist = calculate_dtw_distance(ref_sig.values, test_sig.values)
    # print("DTW distance:", dist)

    print(" All data loaded successfully.")


if __name__ == "__main__":
    main()
