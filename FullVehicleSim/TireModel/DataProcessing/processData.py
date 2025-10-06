import polars as pl
import os
from pathlib import Path

def process_dat_files():
    # Required columns (case insensitive)
    required_columns = ['V', 'N', 'SA', 'IA', 'RL', 'RE', 'P', 'FX', 'FY', 'FZ', 'MX', 'MZ', 'NFX', 'NFY', 'RST', 'TSTI', 'TSTC', 'TSTO', 'AMBTMP', 'SR']
    
    # Path to RawData folder
    raw_data_path = Path("RawData")
    
    # Get all subdirectories
    subdirs = [d for d in raw_data_path.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        print(f"Processing folder: {subdir.name}")
        
        # Get all .dat files in this subdirectory
        dat_files = list(subdir.glob("*.dat"))
        
        if not dat_files:
            print(f"No .dat files found in {subdir.name}")
            continue
            
        combined_data = []
        
        for dat_file in dat_files:
            print(f"Processing file: {dat_file.name}")
            
            try:
                # Read the file, skip first line (header), use second line as column names
                with open(dat_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 3:  # Need at least header, column names, and one data row
                    print(f"File {dat_file.name} has insufficient data")
                    continue
                
                # Get column headers from second line
                header_line = lines[1].strip()
                columns = [col.strip() for col in header_line.split('\t')]
                
                # Read data starting from third line
                data_lines = [line.strip() for line in lines[2:] if line.strip()]
                
                if not data_lines:
                    print(f"No data rows in {dat_file.name}")
                    continue
                
                # Parse data rows
                data_rows = []
                for line in data_lines:
                    values = [val.strip() for val in line.split('\t')]
                    if len(values) == len(columns):
                        data_rows.append(values)
                
                if not data_rows:
                    print(f"No valid data rows in {dat_file.name}")
                    continue
                
                # Create DataFrame
                df = pl.DataFrame(data_rows, schema=columns, orient="row")
                
                # Create output DataFrame with required columns
                output_data = {}
                
                # Create case-insensitive column mapping
                column_mapping = {}
                for col in columns:
                    column_mapping[col.upper()] = col
                
                for req_col in required_columns:
                    if req_col.upper() in column_mapping:
                        actual_col = column_mapping[req_col.upper()]
                        output_data[req_col] = df[actual_col]
                    else:
                        # Column doesn't exist, fill with -1
                        print(f"Missing column '{req_col}' in file {dat_file.name}")
                        output_data[req_col] = [-1] * len(df)
                
                # Create output DataFrame
                output_df = pl.DataFrame(output_data)
                combined_data.append(output_df)
                
            except Exception as e:
                print(f"Error processing {dat_file.name}: {e}")
                continue
        
        # Combine all DataFrames from this subdirectory
        if combined_data:
            final_df = pl.concat(combined_data, how="vertical")
            
            # Save to CSV
            output_filename = f"{subdir.name}.csv"
            final_df.write_csv(output_filename)
            print(f"Created {output_filename} with {len(final_df)} rows")
        else:
            print(f"No data to save for {subdir.name}")

if __name__ == "__main__":
    process_dat_files()