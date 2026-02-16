import pandas as pd

# Load phenotype
pheno = pd.read_csv("phenotype_all.tsv", sep="\t")

# Drop rows with missing IDs
pheno = pheno.dropna(subset=["FID", "IID"])

# Keep only unique individuals
pheno = pheno.drop_duplicates(subset=["FID","IID"])

# Load genotype FAM
fam = pd.read_csv(
    "TWB_local_CD_preQC.fam",
    sep=r"\s+",
    header=None,
    names=["FID","IID","PAT","MAT","SEX","PHENOTYPE"]
)

pheno_ids = set(zip(pheno.FID.astype(str), pheno.IID.astype(str)))
geno_ids  = set(zip(fam.FID.astype(str), fam.IID.astype(str)))

common = pheno_ids.intersection(geno_ids)
only_pheno = pheno_ids - geno_ids
only_geno  = geno_ids - pheno_ids

print("Total phenotype individuals:", len(pheno_ids))
print("Total genotype individuals:", len(geno_ids))
print("Overlap (predictable):", len(common))
print("Only in phenotype (no genotype):", len(only_pheno))
print("Only in genotype (no phenotype):", len(only_geno))

# Save lists for documentation
pd.DataFrame(list(only_pheno), columns=["FID","IID"]).to_csv("pheno_no_genotype.csv", index=False)
pd.DataFrame(list(only_geno), columns=["FID","IID"]).to_csv("geno_no_pheno.csv", index=False)
