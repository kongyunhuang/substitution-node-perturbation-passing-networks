"""
Empirical mean positions of the ten playing roles (layout for Figure 2).

Mean pass-endpoint location per role (passer at origin, receiver at destination), stretched
per axis to fill the pitch with order and spacing preserved, then lightly de-overlapped in
the projected space used by Figure 2a.
Inputs: data/events_pass.parquet, data/player_positions.parquet
Outputs: results/tables/role_positions_empirical.npz (roles, xy_raw, xy_deoverlap),
         results/figures/_design_drafts/role_pos_compare.png (three-panel comparison)
Run: python code/figures/fig2_role_positions.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ROLES = ["GK","RB","CB","LB","DM","CM","AM","RW","LW","ST"]
ROLE_MAP = {
    "Goalkeeper":"GK","Right Back":"RB","Right Wing Back":"RB",
    "Right Center Back":"CB","Left Center Back":"CB","Center Back":"CB",
    "Left Back":"LB","Left Wing Back":"LB",
    "Right Defensive Midfield":"DM","Left Defensive Midfield":"DM","Center Defensive Midfield":"DM",
    "Right Center Midfield":"CM","Left Center Midfield":"CM",
    "Center Attacking Midfield":"AM","Right Attacking Midfield":"AM","Left Attacking Midfield":"AM",
    "Right Wing":"RW","Right Midfield":"RW","Left Wing":"LW","Left Midfield":"LW",
    "Center Forward":"ST","Right Center Forward":"ST","Left Center Forward":"ST"}
RIDX = {r:i for i,r in enumerate(ROLES)}

passes = pd.read_parquet(DATA/"events_pass.parquet")
net = passes[passes.outcome.isna() & ~passes.is_set_piece].copy()
pp = pd.read_parquet(DATA/"player_positions.parquet")
pos = {(m,p): ROLE_MAP.get(q) for m,p,q in zip(pp.match_id,pp.player_id,pp.position)}

# accumulate endpoints per role: passer at origin, receiver at destination
acc = {r: [np.zeros(2), 0] for r in ROLES}
rp = net.apply(lambda r: pos.get((r.match_id, r.player_id)), axis=1)
rr = net.apply(lambda r: pos.get((r.match_id, r.recipient_id)), axis=1)
for role, x, y in zip(rp, net.x.values, net.y.values):
    if role: acc[role][0] += [x,y]; acc[role][1]+=1
for role, x, y in zip(rr, net.end_x.values, net.end_y.values):
    if role: acc[role][0] += [x,y]; acc[role][1]+=1

emp = {}
print("role  empirical mean (x,y)   n_endpoints   (template x,y)")
TEMPLATE = {"GK":(8,40),"RB":(30,66),"CB":(24,40),"LB":(30,14),"DM":(45,40),
            "CM":(57,40),"AM":(69,40),"RW":(74,62),"LW":(74,18),"ST":(88,40)}
for r in ROLES:
    m = acc[r][0]/max(acc[r][1],1)
    emp[r] = m
    print(f"  {r:>3}  ({m[0]:5.1f},{m[1]:5.1f})  n={acc[r][1]:>7}   template({TEMPLATE[r][0]},{TEMPLATE[r][1]})")

# calibrate: stretch per axis to the target pitch range, then de-overlap in projected space
SHX, SHY, GAP = 0.42, 0.40, 50.0
def proj1(x, y):
    yd = 80.0 - y; return x + SHX*yd, SHY*yd + GAP
def invproj1(sx, sy):
    yd = (sy - GAP)/SHY; return sx - SHX*yd, 80.0 - yd
raw = np.array([emp[r] for r in ROLES], float)
# per-axis linear map, order and spacing preserved
TX0, TX1, TY0, TY1 = 12.0, 93.0, 14.0, 66.0
tx = (raw[:,0]-raw[:,0].min())/(raw[:,0].max()-raw[:,0].min())
ty = (raw[:,1]-raw[:,1].min())/(raw[:,1].max()-raw[:,1].min())
cal = np.column_stack([TX0 + tx*(TX1-TX0), TY0 + ty*(TY1-TY0)])
# de-overlap: small repulsion radius, strong spring back to the anchor
scr = np.array([proj1(x, y) for x, y in cal]); anchor = scr.copy()
R = 11.0
for it in range(600):
    disp = np.zeros_like(scr)
    for i in range(10):
        for j in range(i+1, 10):
            d = scr[i]-scr[j]; dist = np.hypot(*d)
            if dist < R:
                if dist < 1e-6:
                    d = np.array([0.3*(1 if i%2 else -1), 0.3]); dist = np.hypot(*d)
                push = (R-dist)/2 * d/dist
                disp[i]+=push; disp[j]-=push
    scr += disp
    scr += 0.10*(anchor-scr)
xy = np.array([invproj1(sx, sy) for sx, sy in scr])
xy[:,0] = np.clip(xy[:,0], 5, 115); xy[:,1] = np.clip(xy[:,1], 5, 75)
deov = {r: xy[i] for i,r in enumerate(ROLES)}
print("\nafter de-overlap (x,y) and displacement from the empirical mean:")
for i,r in enumerate(ROLES):
    print(f"  {r:>3}  ({xy[i,0]:5.1f},{xy[i,1]:5.1f})   d={np.hypot(*(xy[i]-raw[i])):.1f}")

np.savez(ROOT/"results/tables/role_positions_empirical.npz",
         roles=np.array(ROLES), xy_raw=raw, xy_deoverlap=xy)

# comparison: template / raw empirical / de-overlapped
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (title, P) in zip(axes, [("template (current)", TEMPLATE),
                                  ("empirical (raw)", emp), ("empirical + de-overlap", deov)]):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="white", line_color="#888", linewidth=1)
    pitch.draw(ax=ax)
    for r in ROLES:
        x,y = P[r]
        ax.scatter([x],[y], s=420, c="#F0A080", edgecolor="#333", zorder=5)
        ax.text(x, y, r, ha="center", va="center", fontsize=7, zorder=6, weight="bold")
    ax.set_title(title, fontsize=11)
    ax.annotate("attack", xy=(105,-5), xytext=(75,-5),
                arrowprops=dict(arrowstyle="-|>", color="#888"), fontsize=8, color="#888")
fig.savefig(ROOT/"results/figures/_design_drafts/role_pos_compare.png",
            dpi=140, bbox_inches="tight")
print("saved role_pos_compare.png (3-panel)")
