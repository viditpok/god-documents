from pdfminer.high_level import extract_text

files = [
    "/Users/devang/Desktop/Ai in Fin/Capstone/project/MGT6785_Project_Proposal 2.pdf",
    "/Users/devang/Desktop/Ai in Fin/Capstone/project/20220727_NLP_Statements_Research_Note_Final 2.pdf",
    "/Users/devang/Desktop/Ai in Fin/Capstone/project/2410.14841v1 2.pdf",
    "/Users/devang/Desktop/Ai in Fin/Capstone/project/1340005_Regime_Based_Factor_Allocation 2.pdf",
]

output_file = "/Users/devang/Desktop/Ai in Fin/Capstone/project/extracted_text.txt"

with open(output_file, "w") as out:
    for f in files:
        out.write(f"\n\n--- START {f} ---\n")
        try:
            text = extract_text(f)
            # Write full text to file
            out.write(text)
            out.write(f"\n--- END {f} ---\n")
            print(f"Successfully extracted {f}")
        except Exception as e:
            out.write(f"\nError reading {f}: {e}\n")
            print(f"Error extracting {f}: {e}")

print(f"Done. Output written to {output_file}")
