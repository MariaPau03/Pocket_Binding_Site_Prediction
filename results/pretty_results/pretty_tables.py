import os
import sys
import pandas as pd

if len(sys.argv) < 2 or len(sys.argv) > 4:
    print("Usage: python pretty_tables.py path/to/file.csv [output.html] [output.md]")
    print("  - If output paths are omitted, writes file1.html and file1.md next to the CSV.")
    sys.exit(1)

csv_path = sys.argv[1]
output_html = sys.argv[2] if len(sys.argv) >= 3 else None
output_md = sys.argv[3] if len(sys.argv) == 4 else None

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"Input CSV not found: {csv_path}")

# Read CSV as generic table
df = pd.read_csv(csv_path)

# Clean header: trim whitespace
df.columns = [str(c).strip() for c in df.columns]

# Optional: if there is a numeric score column, round to 2 decimals
score_candidate = None
for candidate in ["score", "pocket_id", "residue_id"]:
    if candidate in (col.lower() for col in df.columns):
        score_candidate = candidate
        break

if score_candidate is not None and score_candidate.lower() in (col.lower() for col in df.columns):
    # Safely format numeric-like columns when possible (not all columns should be forced)
    col_name = next(col for col in df.columns if col.lower() == score_candidate.lower())
    if pd.api.types.is_numeric_dtype(df[col_name]):
        df[col_name] = df[col_name].astype(float).map("{:.2f}".format)

base_name = os.path.splitext(os.path.basename(csv_path))[0]
# Default output directory is results/pretty_results from project root
default_output_dir = os.path.join(os.getcwd(), "results", "pretty_results")
if not os.path.exists(default_output_dir):
    os.makedirs(default_output_dir, exist_ok=True)

base_dir = default_output_dir

default_base = "aa_in_pocket_site"
if output_html is None:
    output_html = os.path.join(base_dir, f"{default_base}_{base_name}.html")
if output_md is None:
    output_md = os.path.join(base_dir, f"{default_base}_{base_name}.md")

# Write pretty HTML and Markdown
with open(output_html, "w", encoding="utf-8") as f_html:
    f_html.write(df.to_html(index=False, classes="table table-striped", border=0, justify="center"))

with open(output_md, "w", encoding="utf-8") as f_md:
    f_md.write(df.to_markdown(index=False))

print(f"Written pretty table to: {output_html} and {output_md} (total rows: {len(df)})")