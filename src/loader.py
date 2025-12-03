import os
import glob
import pandas as pd


# =========================
# Load a single signature TSV
# =========================
def load_signature_tsv(path):
    """
    Load a single online signature file (.tsv) into a pandas DataFrame.
    Columns include:
    t, x, y, pressure, penup, azimuth, inclination
    """
    df = pd.read_csv(path, sep="\t")
    return df


# =========================
# Load writers list
# =========================
def load_writers(data_dir):
    """
    Reads writers.tsv and returns list of writer IDs as strings (e.g., '001')
    """
    writers_path = os.path.join(data_dir, "writers.tsv")
    writers_df = pd.read_csv(writers_path, sep="\t", header=None)
    writers = writers_df[0].astype(int).astype(str).str.zfill(3).tolist()
    return writers


# =========================
# Load ground truth labels
# =========================
def load_ground_truth(data_dir):
    """
    Reads gt.tsv, returning a dictionary:
    {
        '001-01': 'genuine',
        '001-02': 'forgery',
        ...
    }
    """
    gt_path = os.path.join(data_dir, "gt.tsv")
    df = pd.read_csv(gt_path, sep="\t", header=None)
    df.columns = ["signature_id", "label"]

    return dict(zip(df["signature_id"], df["label"]))


# =========================
# Load enrollment signatures for a writer
# =========================
def load_enrollment_signatures(writer_id, data_dir):
    """
    Returns a list of 5 DataFrames for the writer's enrollment signatures.
    Filenames like: 001-g-01.tsv ... 001-g-05.tsv
    """
    enroll_dir = os.path.join(data_dir, "enrollment")
    patterns = sorted(glob.glob(os.path.join(enroll_dir, f"{writer_id}-g-*.tsv")))

    signatures = [load_signature_tsv(p) for p in patterns]
    return signatures


# =========================
# Load verification signatures + their labels
# =========================
def load_verification_signatures(writer_id, data_dir, gt_dict):
    """
    Returns a list of dicts:
    [
        {
            "id": "001-01",
            "df": DataFrame,
            "label": "genuine" or "forgery"
        },
        ...
    ]
    """

    ver_dir = os.path.join(data_dir, "verification")
    patterns = sorted(glob.glob(os.path.join(ver_dir, f"{writer_id}-*.tsv")))

    results = []

    for path in patterns:
        filename = os.path.basename(path).replace(".tsv", "")  # e.g. 001-01
        df = load_signature_tsv(path)
        label = gt_dict.get(filename, None)

        results.append({
            "id": filename,
            "df": df,
            "label": label
        })

    return results


# =========================
# Load entire dataset
# =========================
def load_all_data(data_dir="data"):
    """
    Loads everything and returns:

    {
        "001": {
            "enrollment": [df1..df5],
            "verification": [
                {"id":..., "df":..., "label":...},
                ...
            ]
        },
        ...
    }
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found!")

    print(f"Loading data from: {os.path.abspath(data_dir)}")

    writers = load_writers(data_dir)
    gt_dict = load_ground_truth(data_dir)

    dataset = {}

    for writer_id in writers:
        print(f" → Loading writer {writer_id}")

        dataset[writer_id] = {
            "enrollment": load_enrollment_signatures(writer_id, data_dir),
            "verification": load_verification_signatures(writer_id, data_dir, gt_dict),
        }

    print("All writers loaded!")
    return dataset
