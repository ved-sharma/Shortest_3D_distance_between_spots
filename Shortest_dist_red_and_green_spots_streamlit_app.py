import re
import numpy as np
import pandas as pd
from scipy.spatial import distance
import streamlit as st

st.set_page_config(
    page_title="Shortest Distance Calculator", layout="centered"
)

st.title("Shortest 3D Distance Calculator")

# Instruction Message Block
st.info(
    "This app takes the Imaris-generated ...Position.csv files for two sets of spots (red and green) and "
    "calculates the shortest 3D Euclidean distances between them for each cell. \n\n"
    "Cells are identified by their unique values in columns **CellID** and **Original Image Name**.\n\n"
    '- Ensure that one of the csv files contains the words - **"green"**, and **"position"**.\n' 
    '- The other csv file should contain the words - **"red"**, and **"position"**.\n'
    '- Each csv file should have the following columns: **Position X, Position Y, Position Z, CellID, Original Image Name**.\n'
    '- The app will compute the shortest distances from green to red and from red to green, and provide downloadable CSV files with the results.',
    title="Instructions:"
)

with st.bottom:
    st.caption("Created by: Ved Sharma | Bio-Imaging Resource Center, The Rockefeller University (2026)")

def clean_dataframe(df):
    # Extract first 3 columns (X, Y, Z coordinates)
    first_three_cols = list(df.columns[:3])
    use_cols = first_three_cols + ["CellID", "Original Image Name"]
    df = df[use_cols].copy()

    # Clean "Original Image Name" column
    df["Original Image Name"] = (
        df["Original Image Name"]
        .astype(str)
        .str.replace(r"\[.*?\]$", "", regex=True)
    )
    df["Original Image Name"] = df["Original Image Name"].str.replace(
        r"_R3D_.*$", "", regex=True
    )
    return df


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

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(
        columns=["CellID", "Original Image Name", dist_col_name]
    )


# File Uploader component
uploaded_files = st.file_uploader(
    "Upload ...Position.csv Files (Select both green and red)",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    green_file = None
    red_file = None

    for file in uploaded_files:
        filename_lower = file.name.lower()
        if "green" in filename_lower and "position" in filename_lower:
            green_file = file
        elif "red" in filename_lower and "position" in filename_lower:
            red_file = file

    if not green_file or not red_file:
        st.error(
            "Error: Please upload exactly one CSV containing 'green' and 'position', and one containing 'red' and 'position'."
        )
    else:
        st.success(
            f"Loaded: `{green_file.name}` and `{red_file.name}`", icon="✅"
        )

        if st.button("Calculate Shortest Distances", type="primary"):
            with st.spinner("Processing coordinate data..."):
                # Read CSVs skipping the first 3 rows
                df_green_raw = pd.read_csv(green_file, skiprows=3)
                df_red_raw = pd.read_csv(red_file, skiprows=3)

                df_green = clean_dataframe(df_green_raw)
                df_red = clean_dataframe(df_red_raw)

                # Compute distances
                df_g2r = calculate_nearest_distances(
                    df_green, df_red, "Shortest dist of Green to Red"
                )
                df_r2g = calculate_nearest_distances(
                    df_red, df_green, "Shortest dist of Red to Green"
                )

                st.subheader("Results Preview")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Green to Red**")
                    st.dataframe(df_g2r.head())
                    st.download_button(
                        label="Download Green to Red CSV",
                        data=df_g2r.to_csv(index=False),
                        file_name="shortest_distances_green_to_red.csv",
                        mime="text/csv",
                    )

                with col2:
                    st.write("**Red to Green**")
                    st.dataframe(df_r2g.head())
                    st.download_button(
                        label="Download Red to Green CSV",
                        data=df_r2g.to_csv(index=False),
                        file_name="shortest_distances_red_to_green.csv",
                        mime="text/csv",
                    )