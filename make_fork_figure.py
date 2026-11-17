import os, re, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FOLDER=r"C:\Users\efeka\OneDrive\Desktop\Columbia Thesis"; T=10
p=lambda f:os.path.join(FOLDER,f)
def bblnorm(s): return re.sub(r"\D","",str(s).replace(".0",""))
df=pd.read_csv(p("rentals_owner.csv"),dtype=str,low_memory=False)
df["BBL"]=df["BBL"].map(bblnorm); df["ownername"]=df["ownername"].fillna(df["BBL"])
lot=df.drop_duplicates("BBL").copy(); lot=lot[lot["fingerprint"].notna()].copy()
lot["true_sz"]=lot.groupby("fingerprint")["BBL"].transform("nunique")
lot["app_sz"]=lot.groupby("ownername")["BBL"].transform("nunique")
lot["grp"]=lot.apply(lambda r:"small" if r.true_sz<T else ("hidden" if r.app_sz<T else "visible"),axis=1)
lot["llc"]=lot["ownername"].str.upper().str.contains("LLC")
ac=pd.read_csv(p("acris_sales_clean.csv"),dtype=str,low_memory=False)
ac["BBL"]=ac["BBL"].map(bblnorm); ac["yr"]=pd.to_datetime(ac["date"],errors="coerce").dt.year
acq=ac.dropna(subset=["yr"]).groupby("BBL")["yr"].max().astype(int)
lot=lot.merge(acq.rename("acq"),left_on="BBL",right_index=True,how="inner")
haz=pd.read_csv(p("hpd_haz_byyear.csv")); haz["BBL"]=haz["bbl"].map(bblnorm)
haz["year"]=pd.to_numeric(haz["yr"],errors="coerce"); haz=haz[haz.year>=2010]
idx={(b,int(y)):v for (b,y),v in haz.groupby(["BBL","year"])["n"].sum().items()}
YMIN,YMAX=int(haz.year.min()),int(haz.year.max())
def path(mask):
    out={}
    for r in lot[mask][["BBL","acq"]].to_dict("records"):
        by=r["acq"]-1
        if not(YMIN<=by<=YMAX): continue
        base=np.log1p(idx.get((r["BBL"],by),0))
        for t in range(-4,5):
            cy=r["acq"]+t
            if YMIN<=cy<=YMAX: out.setdefault(t,[]).append(np.log1p(idx.get((r["BBL"],cy),0))-base)
    ts=list(range(-4,5))
    return ts,[np.mean(out[t]) for t in ts],[np.std(out[t])/np.sqrt(len(out[t])) for t in ts],len(out[-1])

RED,BLUE="#c0392b","#2e6da4"; INK="#222222"; MUT="#666666"
tsH,mH,seH,nH=path((lot.grp=="hidden")&lot.llc)
tsS,mS,seS,nS=path((lot.grp=="small")&lot.llc)

fig,ax=plt.subplots(figsize=(10,6.2))
# post-purchase shading + baseline lines
ax.axvspan(-0.5,4.5,color="#f2f2f2",zorder=0)
ax.axhline(0,color=MUT,lw=1,ls="--",zorder=1)
ax.axvline(-0.5,color=MUT,lw=1.2,ls=":",zorder=1)
# series (square=hidden, circle=small -> identity not color-alone)
ax.errorbar(tsH,mH,yerr=[1.96*s for s in seH],marker="s",ms=7,lw=2.4,capsize=3,color=RED,zorder=4)
ax.errorbar(tsS,mS,yerr=[1.96*s for s in seS],marker="o",ms=7,lw=2.4,capsize=3,color=BLUE,zorder=4)
# direct labels at right end
ax.text(4.15,mH[-1],"Hidden-big owner",color=RED,fontweight="bold",va="center",fontsize=11)
ax.text(4.15,mS[-1],"Small owner",color=BLUE,fontweight="bold",va="center",fontsize=11)
# before/after region labels
ax.text(-2.25,0.25,"BEFORE purchase",color=MUT,fontsize=10,ha="center",style="italic")
ax.text(2.0,0.25,"AFTER purchase",color=MUT,fontsize=10,ha="center",style="italic")
ax.text(-0.5,-0.245,"  purchase",color=MUT,fontsize=9,ha="left")
# plain-language takeaways on the plateaus
ax.annotate("keeps getting worse\n(+0.20, ~22% more)",xy=(3,0.20),xytext=(1.15,0.115),
            color=RED,fontsize=9.5,ha="left",va="center",
            arrowprops=dict(arrowstyle="->",color=RED,lw=1.2))
ax.annotate("levels off\n(+0.05)",xy=(3,0.045),xytext=(1.4,-0.02),
            color=BLUE,fontsize=9.5,ha="left",va="center",
            arrowprops=dict(arrowstyle="->",color=BLUE,lw=1.2))
# axes
ax.set_xlim(-4.4,5.6); ax.set_ylim(-0.26,0.30)
ax.set_xticks(range(-4,5))
ax.set_xlabel("Years relative to purchase",fontsize=11)
ax.set_ylabel("Hazardous violations vs. the year before purchase\n(log points;  higher = worse)",fontsize=10.5)
ax.set_title("Figure 6. Before purchase both groups are on the same worsening path;\nafter purchase, hidden-big buildings keep deteriorating while small-owner ones level off",
             fontsize=11.5,fontweight="bold",loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors=INK); ax.grid(axis="y",color="#e6e6e6",lw=0.8,zorder=0)
ax.text(0.0,-0.335,f"Hidden-big LLC lots n={nH:,};  small LLC lots n={nS:,}. Change in ln(1+hazardous B+C violations) vs. the year before the most recent sale.",
        transform=ax.get_xaxis_transform(),fontsize=8,color=MUT)
fig.tight_layout()
fig.savefig(p("fork_hidden_vs_small_hazard.png"),dpi=140,bbox_inches="tight")
import shutil; shutil.copy(p("fork_hidden_vs_small_hazard.png"),"/home/claude/fork_v2.png")
print("done")
