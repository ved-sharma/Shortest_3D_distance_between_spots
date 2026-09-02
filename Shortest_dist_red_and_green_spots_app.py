import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
from scipy.spatial import distance


def process_vesicles(folder_path):
    # Filter CSV files excluding output files
    csv_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
        and not f.startswith("shortest_distances_")
    ]

    green_pos_files = [
        f
        for f in csv_files
        if "green" in f.lower() and "position" in f.lower()
    ]
    red_pos_files = [
        f
        for f in csv_files
        if "red" in f.lower() and "position" in f.lower()
    ]

    if len(green_pos_files) != 1 or len(red_pos_files) != 1:
        raise ValueError(
            f"Expected exactly 1 'green position' and 1 'red position' file.\n"
            f"Found {len(green_pos_files)} green file(s) and {len(red_pos_files)} red file(s)."
        )

    selected_files = {
        "Green": os.path.join(folder_path, green_pos_files[0]),
        "Red": os.path.join(folder_path, red_pos_files[0]),
    }

    dfs = {}
    for key, filepath in selected_files.items():
        headers = pd.read_csv(filepath, skiprows=3, nrows=0).columns
        first_three_cols = list(headers[:3])
        use_cols = first_three_cols + ["CellID", "Original Image Name"]

        df = pd.read_csv(filepath, skiprows=3, usecols=use_cols)
        df["Original Image Name"] = (
            df["Original Image Name"]
            .astype(str)
            .str.replace(r"\[.*?\]$", "", regex=True)
        )
        df["Original Image Name"] = df["Original Image Name"].str.replace(
            r"_R3D_.*$", "", regex=True
        )
        dfs[key] = df

    def calculate_nearest_distances(source_df, target_df, dist_col_name):
        source_xyz = source_df.columns[:3].tolist()
        target_xyz = target_df.columns[:3].tolist()
        target_groups = dict(
            tuple(target_df.groupby(["Original Image Name", "CellID"]))
        )

        results = []
        for (img_name, cell_id), group in source_df.groupby(
            ["Original Image Name", "CellID"], sort=False
        ):
            source_coords = group[source_xyz].to_numpy(dtype=float)
            if (img_name, cell_id) in target_groups:
                target_coords = target_groups[(img_name, cell_id)][
                    target_xyz
                ].to_numpy(dtype=float)
                min_dists = distance.cdist(
                    source_coords, target_coords, metric="euclidean"
                ).min(axis=1)
            else:
                min_dists = np.full(len(group), np.nan)

            sub_df = pd.DataFrame(
                {
                    "CellID": group["CellID"].values,
                    "Original Image Name": group["Original Image Name"].values,
                    dist_col_name: min_dists,
                }
            )
            results.append(sub_df)

        return (
            pd.concat(results, ignore_index=True)
            if results
            else pd.DataFrame(
                columns=["CellID", "Original Image Name", dist_col_name]
            )
        )

    df_g2r = calculate_nearest_distances(
        dfs["Green"], dfs["Red"], "Shortest dist of Green to Red"
    )
    df_r2g = calculate_nearest_distances(
        dfs["Red"], dfs["Green"], "Shortest dist of Red to Green"
    )

    df_g2r.to_csv(
        os.path.join(folder_path, "Shortest_distances_green_to_red.csv"),
        index=False,
    )
    df_r2g.to_csv(
        os.path.join(folder_path, "Shortest_distances_red_to_green.csv"),
        index=False,
    )


# GUI Application Window
def run_app():
    root = tk.Tk()
    root.withdraw()  # Hide the main window frame

    # Force the file dialog to pop up on top of all other windows
    root.wm_attributes("-topmost", True)

    # Show instruction message box with OK / Cancel options
    intro_message = (
        "This program calculates the shortest distance between green " \
        "and red spots for each cell (identified with CellID).\n\n"
        'You should have:\n' \
        '1. Cell centroid coordinates for the green and red spots ' \
        'as csv files, exported from Imaris.\n'
        '2. One file name should contain the words "green" and "Position". ' \
        'The other file "red and "Position"'
    )

    proceed = messagebox.askokcancel(
        title="Shortest Distance Calculator", message=intro_message, parent=root
    )

    # If user clicks Cancel or closes dialog, exit script
    if not proceed:
        root.destroy()
        return
    
    # Open folder selection dialog
    folder_selected = filedialog.askdirectory(title="Select Folder containing Imaris-generated CSV files", parent=root)

    # Destroy the temporary root window after selection
    root.destroy()

    if not folder_selected:
        return

    try:
        process_vesicles(folder_selected)
        messagebox.showinfo(
            "Shortest Distance Calculator", "DONE!\n\n" \
            "Distance calculations saved in the same folder as:\n" \
            " - Shortest_distances_green_to_red.csv\n" \
            " - Shortest_distances_red_to_green.csv\n\n\n" \
            "Author:\n" \
            "  Ved Sharma\n" \
            "  Bio-Imaging Resource Center, The Rockefeller University\n" \
            "  August 31, 2026"
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    run_app()