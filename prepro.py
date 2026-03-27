import os
from Bio import PDB #leer pdb
from Bio.SeqIO import write #guardar fastas
from Bio.SeqRecord import SeqRecord #sec +nombre
from Bio.Seq import Seq #contiene aa

def pdb_to_fasta(pdb_file, output_dir):
    parser = PDB.PDBParser(QUIET=True) #el lector de pdb
    struct_id = os.path.splitext(os.path.basename(pdb_file))[0]
    structure = parser.get_structure(struct_id, pdb_file)#extaer estructura
    
    ppb = PDB.PPBuilder() #contruye peptidos buscando aa que esten conectados fisicamente en el PDB
    records = []
    
    # Extraer secuencias de todas las cadenas
    for i, chain in enumerate(structure.get_chains()):
        for pp in ppb.build_peptides(chain):
            sequence = pp.get_sequence() #secuencia de aminoacidos de esa cadena
            if len(sequence) > 0: #guardamos nombre y secuencia
                rec = SeqRecord(  
                    sequence, 
                    id=f"{struct_id}_{chain.id}", 
                    description=""
                )
                records.append(rec)
    
    if records: #si encontramos sec, las guardamos en un fasta
        out_path = os.path.join(output_dir, f"{struct_id}.fasta")
        with open(out_path, "w") as f:
            write(records, f, "fasta")
        print(f"FASTA generado: {out_path}")

# Automatización para carpeta
pdb_folder = "data/chen11"
fasta_folder = "data/fastas"
os.makedirs(fasta_folder, exist_ok=True)

for file in os.listdir(pdb_folder):
    if file.endswith(".pdb"):
        pdb_to_fasta(os.path.join(pdb_folder, file), fasta_folder)
