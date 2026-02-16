from pathlib import Path
import argparse
import re
import pandas as pd
import joblib

# ===== helper copied from your pipeline (simplified) =====
def load_plink_raw(raw_path: Path) -> pd.DataFrame:
    """
    Load PLINK .raw produced by `plink --recode A`.
    Strip allele suffixes (_A/_C/_G/_T) and drop PAT/MAT/SEX/PHENOTYPE.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing PLINK raw file: {raw_path}")
    df = pd.read_csv(raw_path, sep=r"\s+", engine="python")
    new_cols = []
    for c in df.columns:
        if c.upper() in {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}:
            new_cols.append(c)
        else:
            new_cols.append(re.sub(r"_[ACGT]+$", "", c, flags=re.IGNORECASE))
    df.columns = new_cols
    for c in ["PAT", "MAT", "SEX", "PHENOTYPE"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    return df
# =========================================================

def load_any_model(path: Path):
    if path.suffix in [".joblib", ".pkl"]:
        return joblib.load(path)
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")

def build_X(df_full: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align new data to what the model expects using model.feature_names_in_.
    In your training script you did df.drop(columns=['FID','IID']) before training,
    so models expect only covariates + ENet + SNPs, not FID/IID. :contentReference[oaicite:1]{index=1}
    """
    if not hasattr(model, "feature_names_in_"):
        raise AttributeError("Model has no feature_names_in_.")
    cols = list(model.feature_names_in_)
    X = df_full.reindex(columns=cols, fill_value=0.0)
    return X

def main():
    ap = argparse.ArgumentParser(description="Predict CD-RISC and RSA using separate CD/RSA PLINK bfiles.")
    ap.add_argument("--cd_raw", required=True,
                    help="PLINK .raw for CD (from TWB_local_CD_preQC).")
    ap.add_argument("--rsa_raw", required=True,
                    help="PLINK .raw for RSA (from TWB_local_RSA_preQC).")
    ap.add_argument("--phenotype", required=True,
                    help="phenotype_all.tsv with FID/IID/AGE/SEX.")
    ap.add_argument("--cd_scorer", required=True,
                    help="ElasticNet scoring model for CD (ENet_CD).")
    ap.add_argument("--rsa_scorer", required=True,
                    help="ElasticNet scoring model for RSA (ENet_RSA).")
    ap.add_argument("--cd_model", required=True,
                    help="Final CD prediction model.")
    ap.add_argument("--rsa_model", required=True,
                    help="Final RSA prediction model.")
    ap.add_argument("--out", default="predictions_cd_rsa_from_phenotype_all.tsv",
                    help="Output TSV file (default: predictions_cd_rsa_from_phenotype_all.tsv).")
    args = ap.parse_args()

    cd_raw_path  = Path(args.cd_raw)
    rsa_raw_path = Path(args.rsa_raw)
    pheno_path   = Path(args.phenotype)

    print(f"Loading CD genotypes from:  {cd_raw_path}")
    geno_cd  = load_plink_raw(cd_raw_path)
    print(f"Loading RSA genotypes from: {rsa_raw_path}")
    geno_rsa = load_plink_raw(rsa_raw_path)

    print(f"Loading phenotype info from: {pheno_path}")
    pheno = pd.read_csv(pheno_path, sep="\t")

    # Standardize covariate names
    if "Age" not in pheno.columns and "AGE" in pheno.columns:
        pheno["Age"] = pheno["AGE"]
    if "Gender" not in pheno.columns and "SEX" in pheno.columns:
        pheno["Gender"] = pheno["SEX"]

    # Keep only rows with FID & IID
    pheno = pheno.dropna(subset=["FID", "IID"]).copy()
    pheno["FID"] = pheno["FID"].astype(str)
    pheno["IID"] = pheno["IID"].astype(str)

    print(f"N rows in phenotype_all with FID/IID: {len(pheno)}")

    # Merge phenotype with CD and RSA genotypes separately
    df_cd  = pheno.merge(geno_cd,  on=["FID", "IID"], how="inner")
    df_rsa = pheno.merge(geno_rsa, on=["FID", "IID"], how="inner")

    print(f"N rows after merging with CD genotypes:  {len(df_cd)}")
    print(f"N rows after merging with RSA genotypes: {len(df_rsa)}")

    if len(df_cd) == 0:
        raise RuntimeError("No matching FID/IID between phenotype_all and CD .raw.")
    if len(df_rsa) == 0:
        raise RuntimeError("No matching FID/IID between phenotype_all and RSA .raw.")

    # Load models
    cd_scorer  = load_any_model(Path(args.cd_scorer))
    rsa_scorer = load_any_model(Path(args.rsa_scorer))
    cd_model   = load_any_model(Path(args.cd_model))
    rsa_model  = load_any_model(Path(args.rsa_model))

    # === CD: compute ENet_CD and predict CD ===
    cd_snp_cols = list(cd_scorer.feature_names_in_)
    for col in cd_snp_cols:
        if col not in df_cd.columns:
            print(f"[CD WARN] SNP {col} not found in CD data; filling with 0.")
            df_cd[col] = 0.0
    X_cd_enet = df_cd[cd_snp_cols].fillna(0.0)
    print("Computing ENet_CD scores...")
    df_cd["ENet_CD"] = cd_scorer.predict(X_cd_enet)

    # Build features for final CD model (drop FID/IID)
    df_cd_feat = df_cd.drop(columns=["FID", "IID"])
    print("Building feature matrix for final CD model...")
    X_cd = build_X(df_cd_feat, cd_model)

    print("Predicting CD...")
    cd_pred = cd_model.predict(X_cd)
    out_cd = df_cd[["FID", "IID"]].copy()
    out_cd["CD_pred"] = cd_pred

    # === RSA: compute ENet_RSA and predict RSA ===
    rsa_snp_cols = list(rsa_scorer.feature_names_in_)
    for col in rsa_snp_cols:
        if col not in df_rsa.columns:
            print(f"[RSA WARN] SNP {col} not found in RSA data; filling with 0.")
            df_rsa[col] = 0.0
    X_rsa_enet = df_rsa[rsa_snp_cols].fillna(0.0)
    print("Computing ENet_RSA scores...")
    df_rsa["ENet_RSA"] = rsa_scorer.predict(X_rsa_enet)

    df_rsa_feat = df_rsa.drop(columns=["FID", "IID"])
    print("Building feature matrix for final RSA model...")
    X_rsa = build_X(df_rsa_feat, rsa_model)

    print("Predicting RSA...")
    rsa_pred = rsa_model.predict(X_rsa)
    out_rsa = df_rsa[["FID", "IID"]].copy()
    out_rsa["RSA_pred"] = rsa_pred

    # === Merge CD and RSA predictions on FID/IID ===
    out = out_cd.merge(out_rsa, on=["FID", "IID"], how="outer")

    out_path = Path(args.out)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Saved predictions to: {out_path}")

if __name__ == "__main__":
    main()
