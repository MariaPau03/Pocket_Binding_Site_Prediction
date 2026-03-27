import numpy as np
import os

def mock_pssm_generator(fasta_file, output_dir): #id desde el fasta
    struct_id = os.path.splitext(os.path.basename(fasta_file))[0]
    output_path = os.path.join(output_dir, f"{struct_id}.pssm")
    
    # 2. Leemos el archivo FASTA que creamos en el paso anterior
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        # 3. Limpiamos el archivo: ignoramos la línea que empieza por ">" 
        # y juntamos todas las letras de la secuencia.
        seq = "".join([l.strip() for l in lines if not l.startswith(">")])
    
    # 4. Creamos el archivo de salida (formato CSV para que sea fácil de leer)
    with open(output_path, "w") as f:
        # Escribimos el encabezado: Índice, Aminoácido y su Score
        f.write("Residue_Index,Residue,Conservation_Score\n")
        
        # 5. Para cada aa en la secuencia:
        for i, aa in enumerate(seq):
            # 6. Generamos un número aleatorio entre 0 y 1.
            # En la vida real, aquí iría el score que te da BLAST. 1-->conservado 2--> poco relevante
            fake_score = np.random.random() 
            #Es aleatorio para que puedas testear la estructura de tu pipeline sin tener que configurar servidores de bioinformática pesados.
            # 7. Guardamos la fila: ej. "1,M,0.8543"
            f.write(f"{i+1},{aa},{fake_score:.4f}\n")

if __name__ == "__main__":
    # All your execution logic stays here
    fasta_folder = "data/fastas"
    pssm_folder = "data/pssms"
    os.makedirs(pssm_folder, exist_ok=True)

    if os.path.exists(fasta_folder):
        for file in os.listdir(fasta_folder):
            if file.endswith(".fasta"):
                input_path = os.path.join(fasta_folder, file)
                mock_pssm_generator(input_path, pssm_folder)
                print(f"PSSM generated for: {file}")
    else:
        print(f"Directory {fasta_folder} not found!")