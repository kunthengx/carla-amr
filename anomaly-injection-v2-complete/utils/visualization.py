from pathlib import Path
import matplotlib.pyplot as plt

def plot_label_distribution(df,label_col,output_path):
    counts=df[label_col].value_counts().sort_index(); fig,ax=plt.subplots(figsize=(7,4)); counts.plot(kind='bar',ax=ax); ax.set_title('Distribusi Label Setelah Anomaly Injection'); ax.set_xlabel('Label (0=Normal, 1=Defect, 2=Theft)'); ax.set_ylabel('Jumlah'); fig.tight_layout(); Path(output_path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(output_path,dpi=160); plt.close(fig)
