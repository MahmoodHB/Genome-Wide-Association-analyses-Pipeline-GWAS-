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

def build_X(df_full: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align new data to what the model expects using model.feature_names_in_.
    In training you did df.drop(columns=['FID','IID']) before training,
    so models expect only covariates + ENet + SNPs, not FID/IID.
    """
    if not hasattr(model, "feature_names_in_"):
        raise AttributeError("Model has no feature_names_in_.")
    cols = list(model.feature_names_in_)
    X = df_full.reindex(columns=cols, fill_value=0.0)
    return X

def nice_pred_colname(model_path: Path) -> str:
    """
    Map model filename to a nice prediction column name, shared for CD & RSA.
    """
    name = model_path.name
    if "RandomForest" in name:
        return "pred_RandomForest"
    if "SVM" in name:
        return "pred_SVM"
    if "ElasticNet_" in name and "Scoring" not in name:
        return "pred_ElasticNet"
    if "XGBoost" in name:
        return "pred_XGBoost"
    if "LightGBM" in name:
        return "pred_LightGBM"
    # fallback
    return "pred_" + model_path.stem

def main():
    ap = argparse.ArgumentParser(description="Predict CD-RISC and RSA for all models.")
    ap.add_argument("--cd_raw", required=True,
                    help="PLINK .raw for CD (from TWB_local_CD_preQC).")
    ap.add_argument("--rsa_raw", required=True,
                    help="PLINK .raw for RSA (from TWB_local_RSA_preQC).")
    ap.add_argument("--phenotype", required=True,
                    help="phenotype_all.tsv with FID/IID/AGE/SEX.")
    ap.add_argument("--models_dir", required=True,
                    help="Directory with saved_models (beagle_*.joblib).")
    ap.add_argument("--out_prefix", default="predictions_from_phenotype_all",
                    help="Prefix for output CSV files.")
    args = ap.parse_args()

    cd_raw_path  = Path(args.cd_raw)
    rsa_raw_path = Path(args.rsa_raw)
    pheno_path   = Path(args.phenotype)
    models_dir   = Path(args.models_dir)

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

    # === Load ElasticNet scoring models (for ENet_CD / ENet_RSA) ===
    cd_scorers  = sorted(models_dir.glob("*ElasticNet_Scoring_CD*.joblib"))
    rsa_scorers = sorted(models_dir.glob("*ElasticNet_Scoring_RSA*.joblib"))

    if not cd_scorers:
        raise RuntimeError("No CD scoring model (*ElasticNet_Scoring_CD*.joblib) found in models_dir.")
    if not rsa_scorers:
        raise RuntimeError("No RSA scoring model (*ElasticNet_Scoring_RSA*.joblib) found in models_dir.")

    cd_scorer  = joblib.load(cd_scorers[0])
    rsa_scorer = joblib.load(rsa_scorers[0])

    # === CD: compute ENet_CD ===
    cd_snp_cols = list(cd_scorer.feature_names_in_)
    for col in cd_snp_cols:
        if col not in df_cd.columns:
            print(f"[CD WARN] SNP {col} not found in CD data; filling with 0.")
            df_cd[col] = 0.0
    X_cd_enet = df_cd[cd_snp_cols].fillna(0.0)
    print("Computing ENet_CD scores...")
    df_cd["ENet_CD"] = cd_scorer.predict(X_cd_enet)

    # === RSA: compute ENet_RSA ===
    rsa_snp_cols = list(rsa_scorer.feature_names_in_)
    for col in rsa_snp_cols:
        if col not in df_rsa.columns:
            print(f"[RSA WARN] SNP {col} not found in RSA data; filling with 0.")
            df_rsa[col] = 0.0
    X_rsa_enet = df_rsa[rsa_snp_cols].fillna(0.0)
    print("Computing ENet_RSA scores...")
    df_rsa["ENet_RSA"] = rsa_scorer.predict(X_rsa_enet)

    # Drop FID/IID for feature matrices
    df_cd_feat  = df_cd.drop(columns=["FID", "IID"])
    df_rsa_feat = df_rsa.drop(columns=["FID", "IID"])

    # ==========================
    # 1. Predict with ALL CD models (and combined file)
    # ==========================
    cd_models = sorted(models_dir.glob("beagle_CD_*.joblib"))
    print(f"\nFound {len(cd_models)} CD models")

    cd_combined = df_cd[["FID", "IID"]].copy()

    for model_path in cd_models:
        print(f"\nRunning CD model: {model_path.name}")
        cd_model = joblib.load(model_path)

        X_cd = build_X(df_cd_feat, cd_model)
        cd_preds = cd_model.predict(X_cd)

        colname = nice_pred_colname(model_path)
        cd_combined[colname] = cd_preds

        # per-model file (row-level)
        out_cd = df_cd[["FID", "IID"]].copy()
        out_cd["CD_pred"] = cd_preds
        out_name = f"{args.out_prefix}_{model_path.stem}_CD_predictions.csv"
        out_path = Path(out_name)
        out_cd.to_csv(out_path, index=False)
        print(f"Saved CD predictions (row-level): {out_path}")

    # Save combined CD file (row-level)
    combined_cd_path = Path(f"{args.out_prefix}_CD_all_models.csv")
    cd_combined.to_csv(combined_cd_path, index=False)
    print(f"\nSaved combined CD predictions (row-level): {combined_cd_path}")

    # NEW: aggregate CD predictions per person (mean over rows with same FID+IID)
    cd_per_person = cd_combined.groupby(["FID", "IID"], as_index=False).mean(numeric_only=True)
    combined_cd_person_path = Path(f"{args.out_prefix}_CD_all_models_per_person.csv")
    cd_per_person.to_csv(combined_cd_person_path, index=False)
    print(f"Saved combined CD predictions (per person): {combined_cd_person_path}")

    # ==========================
    # 2. Predict with ALL RSA models (and combined file)
    # ==========================
    rsa_models = sorted(models_dir.glob("beagle_RSA_*.joblib"))
    print(f"\nFound {len(rsa_models)} RSA models")

    rsa_combined = df_rsa[["FID", "IID"]].copy()

    for model_path in rsa_models:
        print(f"\nRunning RSA model: {model_path.name}")
        rsa_model = joblib.load(model_path)

        X_rsa = build_X(df_rsa_feat, rsa_model)
        rsa_preds = rsa_model.predict(X_rsa)

        colname = nice_pred_colname(model_path)
        rsa_combined[colname] = rsa_preds

        out_rsa = df_rsa[["FID", "IID"]].copy()
        out_rsa["RSA_pred"] = rsa_preds

        out_name = f"{args.out_prefix}_{model_path.stem}_RSA_predictions.csv"
        out_path = Path(out_name)
        out_rsa.to_csv(out_path, index=False)
        print(f"Saved RSA predictions (row-level): {out_path}")

    # Save combined RSA file (row-level)
    combined_rsa_path = Path(f"{args.out_prefix}_RSA_all_models.csv")
    rsa_combined.to_csv(combined_rsa_path, index=False)
    print(f"\nSaved combined RSA predictions (row-level): {combined_rsa_path}")

    # NEW: aggregate RSA predictions per person
    rsa_per_person = rsa_combined.groupby(["FID", "IID"], as_index=False).mean(numeric_only=True)
    combined_rsa_person_path = Path(f"{args.out_prefix}_RSA_all_models_per_person.csv")
    rsa_per_person.to_csv(combined_rsa_person_path, index=False)
    print(f"Saved combined RSA predictions (per person): {combined_rsa_person_path}")

    print("\nAll model predictions completed.")

if __name__ == "__main__":
    main()
