periodic_table_atomz = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "S": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}

def get_atomz(elmtstr):
    return periodic_table_atomz[elmtstr]
