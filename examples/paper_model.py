"""A bounded one-point minimization and a band plot, not a full phase diagram."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loop_current import CurrentModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mesh",type=int,default=24)
    parser.add_argument("--maxiter",type=int,default=32)
    parser.add_argument("--t",type=float,default=1.0)
    parser.add_argument("--interaction",type=float,default=4.0)
    parser.add_argument("--tp",type=float,default=.2)
    parser.add_argument("--filling",type=float,default=.4)
    parser.add_argument("--output",type=Path,default=Path("examples/output"))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    model=CurrentModel(args.mesh,args.interaction,args.tp)
    result=model.minimize(args.t,args.filling,maxiter=args.maxiter)
    (args.output/"one_point.json").write_text(json.dumps(result,indent=2)+"\n")
    bands=model.bands(result["currents"],args.t)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,band,label in zip(axes,bands,["lower band","upper band"]):
        im=ax.pcolormesh(model.kx/np.pi,model.ky/np.pi,band,shading="auto",cmap="viridis")
        ax.set(xlabel="kx / pi",ylabel="ky / pi",title=label)
        fig.colorbar(im,ax=ax,label="energy (same units as t and V)")
    fig.suptitle(f"Four-current ansatz: bounded {args.mesh} x {args.mesh} example, t/V={args.t/args.interaction:g}, filling={args.filling}")
    fig.savefig(args.output/"bands.png",dpi=160);plt.close(fig)
    np.savez_compressed(args.output/"bands.npz",kx=model.kx,ky=model.ky,bands=bands,currents=result["currents"])
    print(args.output/"bands.png")


if __name__=="__main__":main()
